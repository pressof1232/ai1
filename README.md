# anime-watcher

A production-style local automation service for the "watch anime together"
workflow using **VLC** and two **Ollama** models.

- Vision model: `gemma3:4b`
- Companion/text model: `qwen2.5:7b`
- Fully local — no cloud APIs, no Docker.

---

## Overview

`anime-watcher` monitors the folder where VLC saves screenshots and
automatically:

1. Detects new screenshot files (PNG / JPG / JPEG).
2. Waits until the file is fully written.
3. Pre-processes the image (resize, subtitle crop).
4. Skips duplicate or nearly-identical frames.
5. Sends the image to `gemma3:4b` (vision) and extracts structured scene context.
6. Sends a compact scene update to `qwen2.5:7b` (text companion).
7. Stores the internal assistant reaction in SQLite for future use.

The vision output is **internal context** — it is not shown to the user
by default.  The text model's response is designed to be connected later
to TTS, a desktop overlay, or an assistant memory layer.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Ollama | Running locally on `http://127.0.0.1:11434` |
| `gemma3:4b` | Pulled in Ollama |
| `qwen2.5:7b` | Pulled in Ollama |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/pressof1232/ai1
cd ai1
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your configuration

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` and set at minimum:

```yaml
screenshot_folder: "C:/Users/YourName/Pictures/VLC"
```

### 4. Pull the Ollama models (if not already done)

```bash
ollama pull gemma3:4b
ollama pull qwen2.5:7b
```

### 5. Configure VLC to save screenshots

In VLC: **Tools → Preferences → Video → Video snapshots**  
Set the directory to match `screenshot_folder` in your `config.yaml`.

---

## Running

### Watcher mode (normal operation)

```bash
python main.py
# or with explicit config path:
python main.py --config /path/to/config.yaml
```

Press **Ctrl+C** to stop.

### Manual test mode (single image, no watcher)

```bash
python main.py --test path/to/screenshot.png
```

This runs the full vision → text pipeline on one image and prints all
intermediate results to the terminal.  Useful for tuning prompts and
validating the pipeline.

---

## Project structure

```
anime-watcher/
├── main.py                  # CLI entry point
├── requirements.txt
├── pyproject.toml
├── config.example.yaml      # Annotated example configuration
├── README.md
├── data/                    # Created automatically at runtime
│   ├── state.db             # SQLite state database
│   └── temp/                # Temporary preprocessed images
└── src/
    ├── __init__.py
    ├── config.py            # YAML config loading + pydantic validation
    ├── logging_setup.py     # Structured logging setup
    ├── schemas.py           # Pydantic data schemas
    ├── prompts.py           # All model prompts (edit here to tune)
    ├── watcher.py           # Watchdog folder watcher
    ├── stabilizer.py        # File write-stability detection
    ├── preprocessor.py      # Pillow image preprocessing
    ├── deduplication.py     # Perceptual hash deduplication
    ├── vision_client.py     # Ollama gemma3:4b client
    ├── text_client.py       # Ollama qwen2.5:7b client
    ├── state.py             # SQLite state persistence
    ├── pipeline.py          # Orchestration pipeline
    └── test_mode.py         # Manual test mode
```

---

## Configuration reference

See [`config.example.yaml`](config.example.yaml) for fully annotated
settings.  Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `screenshot_folder` | *(required)* | VLC screenshot output folder |
| `temp_folder` | `data/temp` | Temporary file cache |
| `ollama_base_url` | `http://127.0.0.1:11434` | Ollama API URL |
| `vision_model` | `gemma3:4b` | Vision model name |
| `text_model` | `qwen2.5:7b` | Companion text model name |
| `preprocessing.resize_full_frame` | `true` | Resize frames before sending |
| `preprocessing.full_frame_width` | `1280` | Target width in pixels |
| `preprocessing.crop_subtitle_region` | `true` | Also send subtitle crop |
| `deduplication.hash_distance_threshold` | `5` | pHash distance (0=identical) |
| `deduplication.cooldown_seconds` | `3.0` | Minimum seconds between frames |
| `debug` | `false` | Enable DEBUG log output |

---

## Extending the project

The modular design makes it straightforward to add:

- **Desktop overlay** — consume `state.last_assistant_response` from SQLite
- **TTS** — pipe `AssistantResponse.raw_text` to a TTS engine
- **Hotkeys** — add a hotkey listener that triggers `pipeline.process()`
- **VLC control** — use the VLC HTTP API alongside the watcher
- **Scene history** — store each `VisionResult` row in the SQLite database
- **Periodic analysis** — add a timer task in `main.py` alongside the watcher
- **Push-to-talk / wake-word** — add as an async task in `_run_watcher()`

---

## Known limitations

- Tested primarily on Windows; Linux/macOS should work but signal handling
  differs slightly (see `main.py`).
- The vision model (`gemma3:4b`) must support multimodal input in your
  Ollama version.
- Subtitle detection accuracy depends on the model and subtitle style.
- Very fast screenshot sequences may still saturate the queue; the cooldown
  setting mitigates this.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: Configuration file not found` | Copy `config.example.yaml` → `config.yaml` and set `screenshot_folder` |
| `Ollama unavailable` errors | Ensure `ollama serve` is running |
| Vision model returns non-JSON | Check `prompts.py` and test with `--test` mode |
| Frames processed too frequently | Increase `deduplication.cooldown_seconds` |
| Frames skipped too aggressively | Decrease `deduplication.hash_distance_threshold` |