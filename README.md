# auto-video-lab

A PoC project for exploring automated video editing.
Each script independently applies filler removal, scene splitting, or silence removal to a single input video.

## Setup

```bash
uv sync
```

ffmpeg is required:

```bash
brew install ffmpeg
```

## Scripts

### Filler Removal

Transcribes audio with Whisper and removes filler word segments (e.g. "えー", "あのー").
Supports multi-word matching by combining adjacent words (e.g. "えー" + "と" → "えーと").

**API version** (OpenAI Whisper API):

```bash
# Set OPENAI_API_KEY in .env
uv run python remove_filler.py input.mp4 --output output.mp4
```

**Local version** (faster-whisper):

```bash
uv run python remove_filler_local.py input.mp4 --output output.mp4 --model large-v3
```

Add custom filler words with `--fillers`:

```bash
uv run python remove_filler_local.py input.mp4 --fillers "なんか" "ほら"
```

### Scene Splitting

Detects scene changes using PySceneDetect and splits the video into separate files per scene.

```bash
uv run python detect_scenes.py input.mp4 --output-dir scenes --threshold 27.0
```

### Silence Removal

**Silero VAD version**:

```bash
uv run python remove_silence_vad.py input.mp4 --output output.mp4 --threshold 0.5 --min-silence-ms 700
```

**ffmpeg silencedetect version** (no additional packages required):

```bash
uv run python remove_silence_ffmpeg.py input.mp4 --output output.mp4 --noise-db -30 --min-duration 0.5
```

## Tech Stack

| Feature | Library |
|---|---|
| Filler removal (API) | OpenAI Whisper API (`whisper-1`) |
| Filler removal (local) | faster-whisper (`large-v3`) |
| Scene splitting | PySceneDetect + OpenCV |
| Silence removal (VAD) | Silero VAD (ONNX) |
| Silence removal (ffmpeg) | ffmpeg silencedetect filter |
| Video cutting / concatenation | ffmpeg |
