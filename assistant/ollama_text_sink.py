"""
Ollama text model fallback sink.

Uses qwen2.5:7b (or any configured text_model) via the local Ollama API.
Called only when the user asks a question and AnythingLLM is unavailable/disabled.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx

from assistant.base_sink import AssistantSink, HandoffUnavailableError
from config.loader import AppConfig
from schemas.vision_schema import ContextBundle

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "text_fallback_prompt.txt"


def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "Ты помощник, который помогает пользователю понять происходящее в аниме. "
        "Используй предоставленный контекст сцены. Отвечай кратко и по-русски."
    )


def _build_prompt(user_question: str, context: ContextBundle, system_prompt: str) -> str:
    """Assemble the full prompt including scene context."""
    parts: list[str] = [system_prompt, ""]

    state = context.rolling_state
    if state.scene_summary:
        parts.append(f"## Текущая сцена\n{state.scene_summary}")

    if state.subtitle_summary:
        parts.append(f"## Недавние субтитры\n{state.subtitle_summary}")

    if state.important_entities:
        entities = "\n".join(f"- {e}" for e in state.important_entities[-10:])
        parts.append(f"## Важные детали\n{entities}")

    if context.recent_frames:
        frames_text_parts = []
        for frame in context.recent_frames[-3:]:
            line = f"[{frame.timestamp.strftime('%H:%M:%S')}] {frame.scene}"
            if frame.subtitles:
                line += f" — «{frame.subtitles}»"
            frames_text_parts.append(line)
        parts.append("## Последние кадры\n" + "\n".join(frames_text_parts))

    if context.scene_history_summary:
        parts.append(f"## История сцен\n{context.scene_history_summary}")

    parts.append(f"\n## Вопрос пользователя\n{user_question}")
    return "\n\n".join(parts)


class OllamaTextSink(AssistantSink):
    """Fallback assistant using local Ollama text model."""

    def __init__(self, cfg: AppConfig) -> None:
        self._base_url = cfg.ollama.base_url.rstrip("/")
        self._model = cfg.ollama.text_model
        self._timeout = cfg.ollama.timeout_seconds
        self._max_retries = cfg.ollama.max_retries
        self._retry_delay = cfg.ollama.retry_delay_seconds
        self._system_prompt = _load_system_prompt()

    async def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def query(self, user_question: str, context: ContextBundle) -> str:
        prompt = _build_prompt(user_question, context, self._system_prompt)

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    "assistant.ollama.query: model=%s attempt=%d question=%r",
                    self._model,
                    attempt,
                    user_question[:60],
                )
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}/api/generate",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("response", "").strip()
                    logger.info("assistant.ollama.response_received: len=%d", len(answer))
                    return answer or "Нет ответа от модели."
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "assistant.ollama.failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay)

        raise HandoffUnavailableError(
            f"Ollama text model failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc
