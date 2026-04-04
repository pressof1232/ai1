"""Pydantic schemas for the VLC Anime Assistant pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Cluster context (episode/session identity for memory isolation)
# ---------------------------------------------------------------------------

class ClusterContext(BaseModel):
    """Identifies an episode/session cluster for isolated memory storage."""

    series_name: str = "default"
    episode_label: str = "default"
    session_id: str = "default"

    def __str__(self) -> str:
        return (
            f"series={self.series_name} "
            f"episode={self.episode_label} "
            f"session={self.session_id}"
        )


# ---------------------------------------------------------------------------
# Vision analysis output (structured result from Ollama vision model)
# ---------------------------------------------------------------------------

class VisualAnalysis(BaseModel):
    """Structured result produced by the Ollama vision model for one frame."""

    subtitles: Optional[str] = Field(
        None,
        description="Visible subtitle text exactly as shown on screen, if any.",
    )
    scene: str = Field(
        ...,
        description="Short description of the current scene (1–2 sentences).",
    )
    important: List[str] = Field(
        default_factory=list,
        description="Important visible details: objects, actions, expressions, etc.",
    )
    visible_people_estimate: Optional[int] = Field(
        None,
        description="Rough count of visible characters/people, if determinable.",
    )
    uncertainty: Optional[str] = Field(
        None,
        description="Notes on what was unclear or uncertain in the analysis.",
    )
    image_quality_note: Optional[str] = Field(
        None,
        description="Notes on image quality issues (blur, dark, cropped, etc.).",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_file: Optional[str] = Field(
        None,
        description="Filename of the source screenshot.",
    )
    frame_hash: Optional[str] = Field(
        None,
        description="Perceptual hash of the frame for deduplication.",
    )


# ---------------------------------------------------------------------------
# Rolling scene state (latest snapshot of what is known about the current scene)
# ---------------------------------------------------------------------------

class SceneState(BaseModel):
    """Continuously updated summary of the current scene."""

    scene_summary: str = ""
    important_entities: List[str] = Field(default_factory=list)
    subtitle_summary: str = ""
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Context bundle (assembled for assistant queries)
# ---------------------------------------------------------------------------

class ContextBundle(BaseModel):
    """Everything the assistant needs to answer a user question."""

    rolling_state: SceneState = Field(default_factory=SceneState)
    recent_frames: List[VisualAnalysis] = Field(default_factory=list)
    recent_subtitles: List[str] = Field(default_factory=list)
    scene_history_summary: Optional[str] = None
    query_timestamp: datetime = Field(default_factory=datetime.utcnow)

    def is_empty(self) -> bool:
        return not self.recent_frames and not self.rolling_state.scene_summary
