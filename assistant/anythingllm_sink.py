"""
AnythingLLM assistant sink.

STATUS: Placeholder adapter with a real HTTP integration attempt.

AnythingLLM exposes a REST API on localhost:3001 (default port).
Key endpoints (as of AnythingLLM v1.x):
  POST /api/v1/workspace/{slug}/chat   — send a message to a workspace
  GET  /api/v1/auth                    — verify API key

This sink:
  1. Checks that AnythingLLM is reachable and the API key is valid.
  2. If reachable, sends the user question together with a context preamble.
  3. If unreachable or disabled → raises HandoffUnavailableError so the
     orchestrator falls back to OllamaTextSink automatically.

Configuration (config.yaml):
    anythingllm:
      enabled: true
      base_url: "http://localhost:3001"
      api_key:  "<your-key>"          # Settings → API Keys in AnythingLLM UI
      workspace_slug: "anime"         # slug shown in AnythingLLM workspace URL
      timeout_seconds: 30

If AnythingLLM is not installed or the API key is not set, set enabled: false
and the system will use Ollama text fallback automatically.
"""

from __future__ import annotations

import logging

import httpx

from assistant.base_sink import AssistantSink, HandoffUnavailableError
from config.loader import AppConfig
from schemas.vision_schema import ContextBundle

logger = logging.getLogger(__name__)


def _build_context_preamble(context: ContextBundle) -> str:
    """Format the scene context as a text preamble prepended to the user question."""
    parts: list[str] = []
    state = context.rolling_state

    if state.scene_summary:
        parts.append(f"[Текущая сцена]: {state.scene_summary}")
    if state.subtitle_summary:
        parts.append(f"[Субтитры]: {state.subtitle_summary}")
    if state.important_entities:
        parts.append("[Детали]: " + ", ".join(state.important_entities[-8:]))
    if context.recent_frames:
        last = context.recent_frames[-1]
        parts.append(f"[Последний кадр {last.timestamp.strftime('%H:%M:%S')}]: {last.scene}")
    if context.scene_history_summary:
        parts.append(f"[История]: {context.scene_history_summary[:300]}")

    return "\n".join(parts)


class AnythingLLMSink(AssistantSink):
    """
    Preferred assistant sink — routes queries through AnythingLLM.

    Falls back gracefully by raising HandoffUnavailableError when:
    - anythingllm.enabled is False in config
    - the AnythingLLM server is not reachable
    - the API key is missing or invalid
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._enabled = cfg.anythingllm.enabled
        self._base_url = cfg.anythingllm.base_url.rstrip("/")
        self._api_key = cfg.anythingllm.api_key
        self._workspace_slug = cfg.anythingllm.workspace_slug
        self._timeout = cfg.anythingllm.timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def is_available(self) -> bool:
        """Return True only if enabled AND the API is reachable with a valid key."""
        if not self._enabled or not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/api/v1/auth",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.debug("anythingllm.unavailable: %s", exc)
            return False

    async def query(self, user_question: str, context: ContextBundle) -> str:
        if not await self.is_available():
            raise HandoffUnavailableError("AnythingLLM is not available or not configured")

        preamble = _build_context_preamble(context)
        # Prepend scene context inline so AnythingLLM can see it
        full_message = f"{preamble}\n\n{user_question}" if preamble else user_question

        payload = {
            "message": full_message,
            "mode": "chat",
        }

        logger.info(
            "assistant.anythingllm.query: workspace=%s question=%r",
            self._workspace_slug,
            user_question[:60],
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/v1/workspace/{self._workspace_slug}/chat",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                # AnythingLLM returns { "textResponse": "...", ... }
                answer = data.get("textResponse") or data.get("response") or ""
                logger.info(
                    "assistant.anythingllm.response_received: len=%d", len(answer)
                )
                return answer.strip() or "Нет ответа от AnythingLLM."
        except HandoffUnavailableError:
            raise
        except Exception as exc:
            logger.warning("assistant.anythingllm.error: %s", exc)
            raise HandoffUnavailableError(f"AnythingLLM request failed: {exc}") from exc
