"""Abstract base class for assistant sinks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.vision_schema import ContextBundle


class HandoffUnavailableError(Exception):
    """Raised when a sink is not available and should be bypassed."""


class AssistantSink(ABC):
    """
    Abstract interface for assistant backends.

    The assistant is NEVER called during the vision pipeline.
    It is only called when the user explicitly asks a question
    (answer_user_question in Pipeline).
    """

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if this sink is ready to accept queries."""

    @abstractmethod
    async def query(self, user_question: str, context: ContextBundle) -> str:
        """
        Send a user question with the assembled context bundle to the assistant.

        Raises
        ------
        HandoffUnavailableError
            If the sink is unavailable and a fallback should be tried.
        """
