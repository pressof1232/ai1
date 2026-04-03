"""Pydantic schemas for structured data exchanged between pipeline stages."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VisionResult(BaseModel):
    """Structured output from the vision model (gemma3:4b).

    All fields are parsed from the JSON the model returns.  Optional fields
    may be ``None`` when the model is uncertain or the detail is absent.
    """

    subtitles: Optional[str] = Field(
        None,
        description="Visible subtitle text exactly as seen on screen, or null.",
    )
    scene: str = Field(
        ...,
        description="Short description of the scene.",
    )
    important: list[str] = Field(
        default_factory=list,
        description="List of notable visible details.",
    )
    visible_people_estimate: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated number of visible people, or null.",
    )
    uncertainty: Optional[str] = Field(
        None,
        description="What the model is unsure about, or null.",
    )
    image_quality_note: Optional[str] = Field(
        None,
        description="Note about image quality issues, or null.",
    )


class SceneUpdate(BaseModel):
    """Compact scene update sent to the text model.

    This is derived from :class:`VisionResult` and carries only the
    information relevant for the companion assistant.
    """

    subtitles: Optional[str] = None
    scene: str
    important: list[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None


class AssistantResponse(BaseModel):
    """Internal response from the text model (qwen2.5:7b).

    Stored in the state database; not shown to the user by default.
    """

    raw_text: str = Field(..., description="Raw text response from the model.")
