"""Ollama vision client.

Sends image bytes to the gemma3:4b vision model through the Ollama HTTP API
and returns a validated :class:`~src.schemas.VisionResult`.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import AppConfig
from src.logging_setup import get_logger
from src.prompts import VISION_SYSTEM_PROMPT, VISION_USER_PROMPT
from src.schemas import VisionResult

log: logging.Logger = get_logger(__name__)

_MAX_ATTEMPTS = 3


class VisionClient:
    """Async client for the Ollama vision model.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._base_url = config.ollama_base_url.rstrip("/")
        timeout = httpx.Timeout(
            connect=config.timeouts.connect,
            read=config.timeouts.read,
            write=config.timeouts.write,
            pool=config.timeouts.connect,
        )
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def analyse(
        self,
        full_frame_bytes: bytes,
        subtitle_crop_bytes: Optional[bytes] = None,
    ) -> VisionResult:
        """Send image(s) to the vision model and return structured results.

        Args:
            full_frame_bytes: JPEG bytes of the full frame.
            subtitle_crop_bytes: Optional JPEG bytes of the subtitle crop.

        Returns:
            A validated :class:`VisionResult`.

        Raises:
            httpx.HTTPError: On network or HTTP errors (after retries).
            ValueError: If the model returns unparseable JSON.
        """
        images = [_encode_b64(full_frame_bytes)]
        if subtitle_crop_bytes:
            images.append(_encode_b64(subtitle_crop_bytes))

        log.info("Vision request started (model=%s)", self._config.vision_model)
        raw_text = await self._call_ollama(images)
        log.info("Vision response received (%d chars)", len(raw_text))

        result = _parse_vision_response(raw_text)
        log.info("Vision parse success: scene=%r", result.scene[:60])
        return result

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_ollama(self, images: list[str]) -> str:
        """Call the Ollama /api/chat endpoint with retry logic."""
        payload = {
            "model": self._config.vision_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": VISION_USER_PROMPT,
                    "images": images,
                },
            ],
        }
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error(
                "Vision API HTTP error: %s — %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise
        except httpx.TimeoutException:
            log.warning("Vision API timeout — will retry")
            raise

        data = response.json()
        return data["message"]["content"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _encode_b64(image_bytes: bytes) -> str:
    """Base64-encode image bytes for the Ollama API."""
    return base64.b64encode(image_bytes).decode("ascii")


def _parse_vision_response(raw: str) -> VisionResult:
    """Extract and validate a :class:`VisionResult` from raw model output.

    The model may wrap JSON in markdown fences; this function strips them
    before parsing.

    Args:
        raw: Raw string returned by the model.

    Returns:
        A validated :class:`VisionResult`.

    Raises:
        ValueError: If no valid JSON can be found or it fails validation.
    """
    # Strip optional markdown code fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = cleaned.replace("```", "").strip()

    # Try to find the first JSON object in the response.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in vision response: {raw[:300]!r}")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Vision response JSON parse error: {exc}\nRaw: {raw[:300]!r}"
        ) from exc

    try:
        return VisionResult.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"VisionResult validation error: {exc}\nData: {data}"
        ) from exc
