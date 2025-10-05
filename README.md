<div align="center">
  <img src="logo.png" alt="Kani TTS Logo" width="150"/>

  # Kani TTS

  A high-quality text-to-speech (TTS) system powered by MLX
</div>

## Features

- **High-quality speech synthesis** using the KaniTTS 370M model
- **FastAPI server** with RESTful API endpoints
- **Configurable generation parameters** (temperature, max tokens, etc.)
- **Sliding window decoding** for smooth, continuous audio output

## Architecture

The project consists of three main components:

1. **[main.py](main.py)** - Standalone audio generation script that creates WAV files locally
2. **[server.py](server.py)** - FastAPI server with both complete and streaming audio endpoints
3. **[client.html](client.html)** - Interactive web frontend for real-time audio generation

### Project Structure

```
kani-mlx/
├── main.py              # Standalone TTS generation script
├── server.py            # FastAPI web server
├── client.html          # Web UI frontend
├── config.py            # Configuration and constants
├── requirements.txt     # Python dependencies
├── audio/              # Audio processing modules
│   ├── player.py       # Audio codec and playback
│   └── streaming.py    # Streaming audio writer
└── generation/         # TTS generation modules
    └── generator.py    # Core TTS generator
```

## Installation

### Prerequisites

- Python 3.10+
- macOS 15.0 or higher (for MLX support)
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd kani-mlx
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Standalone Generation (Local WAV File)

Generate audio and save it to a WAV file:

```bash
python main.py
```

This will:
- Generate speech from the prompt in [main.py:28](main.py#L28)
- Save output to `output.wav`
- Display timing metrics for performance analysis

### Option 2: FastAPI Server + Web Interface

1. Start the server:
```bash
python server.py
```

The server will start on `http://localhost:8000`

2. Open the web interface:
```bash
open client.html
```

Or navigate to `http://localhost:8000` in your browser

## API Endpoints

### `POST /tts`
Generate complete audio file (non-streaming)

**Request:**
```json
{
  "text": "Hello world!",
  "temperature": 0.6,
  "max_tokens": 1200,
  "top_p": 0.95,
  "chunk_size": 25,
  "lookback_frames": 15
}
```

**Response:** WAV audio file

### `POST /stream-tts`
Stream audio chunks for immediate playback

**Request:** Same as `/tts`

**Response:** Streaming PCM audio chunks with metadata headers

## Configuration

Edit [config.py](config.py) to customize:

```python
# Audio settings
CHUNK_SIZE = 25 # Frames per streaming iteration (2.0s)
LOOKBACK_FRAMES = 15 # Context frames for continuity (1.2s)

# Generation parameters
TEMPERATURE = 0.6
TOP_P = 0.95
REPETITION_PENALTY = 1.1
MAX_TOKENS = 1200

# Model configuration
MODEL_NAME = "nineninesix/kani-tts-370m-MLX"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
```

## Technical Details

### Streaming Architecture

The system uses a **sliding window decoder** for smooth audio generation:

1. **Chunk Size (25 frames)** - Outputs ~2.0 seconds of new audio per iteration
2. **Lookback Frames (15 frames)** - Includes ~1.2 seconds of context from previous output

### Performance Metrics

Time to First Token (TTFT) on Macbook Air with M2 Silicon and 8GB RAM is around 2-3 sec. Tested on DuckDuckGo browser.

> **Note:** If you experience audio breaks during streaming, try increasing `CHUNK_SIZE` in [config.py](config.py) to buffer more frames per chunk.

## Models

- **TTS Model:** [nineninesix/kani-tts-370m-MLX](https://huggingface.co/nineninesix/kani-tts-370m-MLX)
- **Codec Model:** [nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)

Models are automatically downloaded from Hugging Face on first run.

## Browser Compatibility

The web interface requires a modern browser with support for:
- Web Audio API
- Fetch API with streaming

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
