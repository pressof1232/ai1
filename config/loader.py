"""Config loader — reads config.yaml and exposes a typed AppConfig object."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class PathsConfig(BaseModel):
    screenshots_folder: str = r"C:\Users\User\Pictures\VLC Screenshots"
    temp_folder: str = r"C:\Users\User\AppData\Local\vlc-anime-assistant\temp"
    logs_folder: str = r"C:\Users\User\AppData\Local\vlc-anime-assistant\logs"
    state_db: str = r"C:\Users\User\AppData\Local\vlc-anime-assistant\state.db"
    scene_memory_db: str = r"C:\Users\User\AppData\Local\vlc-anime-assistant\scene_memory.db"
    debug_export_folder: str = ""

    def expanded(self, field: str) -> Path:
        """Return a Path with %ENV_VAR% expansions resolved."""
        return Path(os.path.expandvars(getattr(self, field)))


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    vision_model: str = "gemma3:4b"
    text_model: str = "qwen2.5:7b"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


class AnythingLLMConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:3001"
    api_key: str = ""
    workspace_slug: str = "anime"
    timeout_seconds: int = 30


class AssistantConfig(BaseModel):
    sink: str = "auto"  # "anythingllm" | "ollama" | "auto"

    @field_validator("sink")
    @classmethod
    def _validate_sink(cls, v: str) -> str:
        allowed = {"anythingllm", "ollama", "auto"}
        if v not in allowed:
            raise ValueError(f"assistant.sink must be one of {allowed}, got {v!r}")
        return v


class WatcherConfig(BaseModel):
    poll_interval_seconds: float = 0.5
    stability_checks: int = 3
    supported_extensions: List[str] = Field(default_factory=lambda: ["png", "jpg", "jpeg"])


class PreprocessingConfig(BaseModel):
    max_width: int = 1280
    max_height: int = 720
    subtitle_crop_top_ratio: float = 0.78
    save_temp_images: bool = False


class DedupConfig(BaseModel):
    hash_size: int = 8
    similarity_threshold: int = 5
    subtitle_cooldown_seconds: float = 10.0
    frame_cooldown_seconds: float = 5.0


class SceneMemoryConfig(BaseModel):
    max_frames: int = 30
    max_subtitle_lines: int = 50
    max_scene_summaries: int = 10


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "assistant.log"


class AnimeStorageConfig(BaseModel):
    """Physical filesystem storage for per-episode data (screenshots, JSONL records)."""

    enabled: bool = False
    root_folder: str = ""   # e.g. "A:/AnimeDB" — leave empty to disable


class DebugConfig(BaseModel):
    enabled: bool = False
    print_raw_vision_response: bool = False


class SessionConfig(BaseModel):
    """Active cluster settings for episode/session-aware memory."""

    series_name: str = "default"
    episode_label: str = "default"
    session_id: str = ""
    auto_generate_session_id: bool = True
    parse_from_path: bool = True


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    anythingllm: AnythingLLMConfig = Field(default_factory=AnythingLLMConfig)
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    scene_memory: SceneMemoryConfig = Field(default_factory=SceneMemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    anime_storage: AnimeStorageConfig = Field(default_factory=AnimeStorageConfig)

    def ensure_dirs(self) -> None:
        """Create all configured directories that do not exist yet."""
        dir_fields = [
            "screenshots_folder",
            "temp_folder",
            "logs_folder",
            "debug_export_folder",
        ]
        for field in dir_fields:
            path_str = getattr(self.paths, field)
            if path_str:
                p = Path(os.path.expandvars(path_str))
                p.mkdir(parents=True, exist_ok=True)
        # DB parent dirs
        for db_field in ("state_db", "scene_memory_db"):
            db_path = Path(os.path.expandvars(getattr(self.paths, db_field)))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        # Anime storage root
        if self.anime_storage.enabled and self.anime_storage.root_folder:
            Path(os.path.expandvars(self.anime_storage.root_folder)).mkdir(
                parents=True, exist_ok=True
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> AppConfig:
    """Load AppConfig from a YAML file.  Missing file → uses defaults."""
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
