"""Configuration loading and validation.

Reads a YAML file and exposes a strongly-typed :class:`AppConfig` object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class PreprocessingConfig(BaseModel):
    """Image preprocessing options."""

    resize_full_frame: bool = True
    full_frame_width: int = Field(1280, ge=64, le=7680)
    crop_subtitle_region: bool = True
    subtitle_region_height_fraction: float = Field(0.20, gt=0.0, lt=1.0)


class DeduplicationConfig(BaseModel):
    """Frame and subtitle deduplication options."""

    hash_distance_threshold: int = Field(5, ge=0, le=64)
    cooldown_seconds: float = Field(3.0, ge=0.0)
    subtitle_history_size: int = Field(5, ge=1)


class TimeoutsConfig(BaseModel):
    """HTTP timeout values for Ollama clients."""

    connect: float = Field(10.0, ge=0.1)
    read: float = Field(120.0, ge=1.0)
    write: float = Field(30.0, ge=1.0)


class AppConfig(BaseModel):
    """Root application configuration."""

    screenshot_folder: Path
    temp_folder: Path = Path("data/temp")
    state_db_path: Path = Path("data/state.db")

    ollama_base_url: str = "http://127.0.0.1:11434"
    vision_model: str = "gemma3:4b"
    text_model: str = "qwen2.5:7b"

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    deduplication: DeduplicationConfig = DeduplicationConfig()
    timeouts: TimeoutsConfig = TimeoutsConfig()

    stabilization_poll_interval: float = Field(0.3, ge=0.05)
    stabilization_stable_count: int = Field(3, ge=1)
    stabilization_timeout: float = Field(10.0, ge=1.0)

    debug: bool = False
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "AppConfig":
        """Create temp and data directories if they don't exist."""
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from *path*.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config contains invalid values.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Copy config.example.yaml to config.yaml and edit it."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
