"""Ollama vision model client (gemma3:4b or any configured vision model)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

from config.loader import AppConfig
from schemas.vision_schema import VisualAnalysis

logger = logging.getLogger(__name__)

# Path to vision prompt (relative to project root)
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "vision_prompt.txt"


def _load_vision_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "Опиши этот кадр аниме на русском языке. "
        "Сначала укажи субтитры (если есть), затем сцену, затем важные детали."
    )


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json_block(text: str) -> Optional[str]:
    """Try to extract a JSON block from model output."""
    # Look for ```json ... ``` or bare { ... }
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def _parse_response(raw: str, source_file: Optional[str], frame_hash: Optional[str]) -> VisualAnalysis:
    """
    Parse model response into a VisualAnalysis.
    Falls back to a minimal analysis if JSON cannot be extracted.
    """
    json_str = _extract_json_block(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            data.setdefault("source_file", source_file)
            data.setdefault("frame_hash", frame_hash)
            # scene is required; provide fallback
            if "scene" not in data or not data["scene"]:
                data["scene"] = raw[:300]
            return VisualAnalysis.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("vision.parse_json_failed: %s — using text fallback", exc)

    # Text fallback: put entire response as scene description
    return VisualAnalysis(
        scene=raw.strip()[:500] if raw.strip() else "Описание недоступно",
        source_file=source_file,
        frame_hash=frame_hash,
    )


class OllamaVisionClient:
    """Sends images to the Ollama vision model and returns VisualAnalysis."""

    def __init__(self, cfg: AppConfig) -> None:
        self._base_url = cfg.ollama.base_url.rstrip("/")
        self._model = cfg.ollama.vision_model
        self._timeout = cfg.ollama.timeout_seconds
        self._max_retries = cfg.ollama.max_retries
        self._retry_delay = cfg.ollama.retry_delay_seconds
        self._prompt = _load_vision_prompt()
        self._debug = cfg.debug.print_raw_vision_response

    async def analyze(
        self,
        img: Image.Image,
        source_file: Optional[str] = None,
        frame_hash: Optional[str] = None,
    ) -> VisualAnalysis:
        """Send image to vision model and return structured analysis."""
        image_b64 = _image_to_base64(img)
        payload = {
            "model": self._model,
            "prompt": self._prompt,
            "images": [image_b64],
            "stream": False,
        }

        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    "vision.request.started: model=%s attempt=%d file=%s",
                    self._model,
                    attempt,
                    source_file or "?",
                )
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}/api/generate",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    raw = data.get("response", "")

                if self._debug:
                    logger.debug("vision.raw_response:\n%s", raw)

                result = _parse_response(raw, source_file, frame_hash)
                logger.info(
                    "vision.request.completed: subtitles=%r scene_len=%d",
                    result.subtitles,
                    len(result.scene),
                )
                return result

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "vision.request.failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay)

        raise RuntimeError(f"Ollama vision model failed after {self._max_retries} attempts: {last_exc}") from last_exc
