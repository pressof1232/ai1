"""Ollama text/companion model client.

Sends a compact scene update (derived from :class:`~src.schemas.VisionResult`)
to qwen2.5:7b and returns the model's internal assistant response.

The response is intended for internal use (state storage, TTS, overlay) and
is NOT shown to the user by default.
"""
from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import AppConfig
from src.logging_setup import get_logger
from src.prompts import TEXT_SYSTEM_PROMPT, build_text_user_message
from src.schemas import AssistantResponse, SceneUpdate

log: logging.Logger = get_logger(__name__)

_MAX_ATTEMPTS = 3


class TextClient:
    """Async client for the Ollama text companion model.

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
        # Maintain a short conversation history for context continuity.
        self._history: list[dict[str, str]] = []

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def react_to_scene(self, update: SceneUpdate) -> AssistantResponse:
        """Send a scene update and get an internal assistant reaction.

        Args:
            update: Compact scene context from the vision model.

        Returns:
            An :class:`AssistantResponse` with the model's internal reaction.

        Raises:
            httpx.HTTPError: On network or HTTP errors (after retries).
        """
        user_message = build_text_user_message(
            subtitles=update.subtitles,
            scene=update.scene,
            important=update.important,
            uncertainty=update.uncertainty,
        )

        log.info("Text-model request started (model=%s)", self._config.text_model)
        raw_text = await self._call_ollama(user_message)
        log.info("Text-model response received (%d chars)", len(raw_text))

        return AssistantResponse(raw_text=raw_text)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_ollama(self, user_message: str) -> str:
        """Call the Ollama /api/chat endpoint with conversation history."""
        self._history.append({"role": "user", "content": user_message})

        messages = [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            *self._history,
        ]

        payload = {
            "model": self._config.text_model,
            "stream": False,
            "messages": messages,
        }

        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error(
                "Text API HTTP error: %s — %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            # Remove the message we just appended so history stays consistent.
            self._history.pop()
            raise
        except httpx.TimeoutException:
            log.warning("Text API timeout — will retry")
            self._history.pop()
            raise

        data = response.json()
        assistant_text: str = data["message"]["content"]

        # Keep conversation history bounded to last 10 exchanges.
        self._history.append({"role": "assistant", "content": assistant_text})
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return assistant_text
