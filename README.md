# VLC Anime Assistant

A Windows-first local automation system for anime watching with VLC, Ollama, and AnythingLLM.

Watches a VLC screenshot folder, silently builds a scene memory using a local vision model, and answers user questions about the current episode using stored context — without interrupting playback.

---

## Architecture

```
VLC screenshot folder
        │
        ▼
ScreenshotWatcher        (watchdog, file stability check)
        │
        ▼
ImagePreprocessor        (resize, subtitle crop)
        │
        ▼
Deduplicator             (perceptual hash, subtitle cooldown)
        │  duplicate → skip
        ▼
OllamaVisionClient       (gemma3:4b → structured VisualAnalysis)
        │
        ▼
SceneMemory / ContextStore   ← THE MEMORY LAYER (SQLite, silent)
        │
        │  (assistant stays silent during screenshot processing)
        │
        ▼  ← only triggered by user question
ContextRetriever         (assembles ContextBundle from stored memory)
        │
        ▼
AssistantSink (auto mode)
  ├─► AnythingLLMSink    (preferred, if enabled and reachable)
  └─► OllamaTextSink     (fallback, qwen2.5:7b)
        │
        ▼
Answer returned to user
```

**Key principle:** The vision pipeline runs silently and only updates memory. The assistant is never called on every frame. It responds only when the user explicitly asks something.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with:
  - `ollama pull gemma3:4b` (vision model)
  - `ollama pull qwen2.5:7b` (text fallback model)
- VLC configured to save screenshots to a specific folder
- (Optional) [AnythingLLM Desktop](https://anythingllm.com) with API key

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/pressof1232/ai1.git
cd ai1
pip install -r requirements.txt
```

### 2. Configure

```bash
copy config.example.yaml config.yaml
```

Edit `config.yaml` and set at minimum:

```yaml
paths:
  screenshots_folder: "C:/Users/YourName/Pictures/VLC Screenshots"
```

All other folders are created automatically under `%LOCALAPPDATA%\vlc-anime-assistant\`.

### 3. Configure VLC to take screenshots

In VLC: **Tools → Preferences → Video → Video snapshots**
- Set the folder to match `paths.screenshots_folder` in your config.

Default VLC screenshot hotkey: `Shift+S`

### 4. Start the watcher

```bash
python main.py
```

The watcher will silently process screenshots and update scene memory.

### 5. Ask questions

In a separate terminal (or integrate into your workflow):

```bash
python main.py --query "что сейчас происходит?"
python main.py --query "о чём они только что говорили?"
python main.py --query "что было в предыдущей сцене?"
```

---

## Manual Test Mode

Test the full pipeline on a single image without running the watcher:

```bash
python main.py --test "C:/path/to/screenshot.png"
```

This will:
1. Preprocess the image
2. Run deduplication check
3. Call Ollama vision model
4. Update scene memory
5. Retrieve and print stored context
6. Print full JSON analysis

---

## Configuration Reference

See `config.example.yaml` for all options with comments. Key sections:

| Section | Purpose |
|---|---|
| `paths` | All folder/DB locations (supports `%ENV_VAR%` expansion) |
| `ollama` | Ollama base URL, model names, timeouts, retries |
| `anythingllm` | AnythingLLM integration (set `enabled: true` to activate) |
| `assistant.sink` | `auto` / `anythingllm` / `ollama` |
| `preprocessing` | Resize limits, subtitle crop ratio |
| `dedup` | Hash size, similarity threshold, cooldowns |
| `scene_memory` | Frame/subtitle/history buffer sizes |
| `logging` | Log level, file logging |
| `debug` | Print raw vision responses |

---

## AnythingLLM Integration

### Current status

AnythingLLM integration is implemented as a real HTTP adapter but is **optional and disabled by default**.

To enable:

```yaml
anythingllm:
  enabled: true
  base_url: "http://localhost:3001"
  api_key: "your-api-key-here"       # Settings → API Keys in AnythingLLM UI
  workspace_slug: "anime"            # slug from your workspace URL
```

The `AnythingLLMSink` uses the standard AnythingLLM REST API:
- `GET /api/v1/auth` — health/auth check
- `POST /api/v1/workspace/{slug}/chat` — send message with context

### Auto fallback

When `assistant.sink: "auto"` (the default), the system:
1. Checks if AnythingLLM is reachable
2. Uses AnythingLLM if available
3. Falls back to Ollama text model automatically if not

This means **the system always works**, regardless of AnythingLLM availability.

---

## Project Structure

```
ai1/
├── main.py                        Entry point (watcher / test / query modes)
├── config.example.yaml            Configuration template
├── requirements.txt
├── prompts/
│   ├── vision_prompt.txt          Prompt for vision model (Russian, JSON output)
│   └── text_fallback_prompt.txt   Prompt for text fallback (Russian, companion)
├── config/
│   └── loader.py                  Typed config loading with Pydantic
├── logging_setup/
│   └── logger.py                  Structured logging setup
├── watcher/
│   └── screenshot_watcher.py      Watchdog-based folder monitor + stability check
├── preprocessing/
│   └── image_processor.py         Image resize and subtitle crop
├── dedup/
│   └── deduplicator.py            Perceptual hash deduplication
├── schemas/
│   └── vision_schema.py           Pydantic models: VisualAnalysis, SceneState, ContextBundle
├── vision/
│   └── ollama_vision_client.py    Ollama vision API client with retry
├── memory/
│   ├── context_store.py           SQLite-backed frame/subtitle/state storage
│   ├── scene_memory.py            Memory ingestion logic (silent layer)
│   └── context_retriever.py       Context bundle assembly on user query
├── assistant/
│   ├── base_sink.py               Abstract AssistantSink interface
│   ├── anythingllm_sink.py        AnythingLLM HTTP adapter (optional, preferred)
│   └── ollama_text_sink.py        Ollama text model fallback
├── state/
│   └── local_state.py             SQLite key-value store for pipeline state
├── orchestrator/
│   └── pipeline.py                Wires all stages; handles user queries
└── tests/
    └── manual_test.py             Manual image test mode
```

---

## Known Limitations

1. **Windows path expansion**: `%USERNAME%` in `config.yaml` paths requires Python's `os.path.expandvars()` — works on Windows, not on Linux/macOS.

2. **Vision model JSON reliability**: `gemma3:4b` does not always return clean JSON. The parser falls back to using the raw text as the scene description when JSON extraction fails.

3. **AnythingLLM API version**: The adapter targets the AnythingLLM v1 REST API. If the API changes in future AnythingLLM versions, `anythingllm_sink.py` may need updating.

4. **No interactive UI**: Currently query mode requires running a separate `python main.py --query "..."` command. A future version could add hotkeys, a tray icon, or a web UI.

5. **Single-folder watcher**: Only one screenshot folder is monitored. Multi-folder support would require extending `ScreenshotWatcher`.

6. **Scene summary is heuristic**: The "significant change" detection in `SceneMemory` uses simple word-overlap — not semantic similarity. A future version could use an embedding model.

---

## Next Steps for Full AnythingLLM Integration

1. **Workspace document injection**: Instead of prepending context inline to chat messages, upload scene summaries as documents to the AnythingLLM workspace so it can use them as RAG context. Use the `/api/v1/document/raw-text` endpoint.

2. **Persistent conversation thread**: Maintain a conversation thread ID in `LocalState` so follow-up questions in AnythingLLM have context from previous answers.

3. **MCP server**: Register this project as an MCP (Model Context Protocol) server so AnythingLLM can pull scene context on demand via tool calls.

4. **Workspace auto-creation**: On first run, auto-create the configured AnythingLLM workspace if it doesn't exist.

5. **Real-time subtitle streaming**: Instead of screenshot-based subtitle detection, hook into VLC's telnet/HTTP interface or subtitle output file for lower-latency subtitle capture.

---

## License

MIT