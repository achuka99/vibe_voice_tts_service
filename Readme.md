# VibeVoice-Realtime FastAPI Backend

Production-ready FastAPI backend for VibeVoice-Realtime text-to-speech with Docker support.

## Features

- 🚀 FastAPI REST API with OpenAPI documentation
- 🔄 WebSocket support for real-time streaming
- 🐳 Fully Dockerized with GPU support
- 📊 Health check endpoints
- 🎙️ Multiple speaker voices
- ⚡ ~200ms first audio latency
- 📝 Comprehensive error handling

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA support (T4 or better recommended)
- At least 4GB GPU memory
- **No manual model download needed!** The Docker container handles this automatically

### 1. Clone and Setup

```bash
# Create project directory
mkdir vibevoice-api && cd vibevoice-api

# Copy all provided files:
# - main.py
# - Dockerfile
# - docker-compose.yml
# - requirements.txt
# - .env.example

# Create environment file
cp .env.example .env

# Edit .env if needed (optional - defaults work fine)
# Set AUTO_DOWNLOAD_MODEL=true to enable automatic model downloading
```

### 2. Build and Run (Model Downloads Automatically!)

```bash
# Build Docker image
docker-compose build

# Start the service (model will download on first start)
docker-compose up -d

# Watch the logs to see model download progress
docker-compose logs -f

# First startup takes 5-10 minutes to download the model (~2GB)
# Subsequent starts are instant!
```

### 3. Test the API

The API will be available at `http://localhost:8000`

View interactive documentation: `http://localhost:8000/docs`

## API Endpoints

### REST API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Text-to-Speech (Non-streaming)
```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the VibeVoice realtime text to speech system.",
    "speaker_name": "Carter"
  }'
```

#### Text-to-Speech (Streaming)
```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This will stream audio in real-time.",
    "speaker_name": "Carter",
    "stream": true
  }' \
  --output speech.wav
```

#### List Available Speakers
```bash
curl http://localhost:8000/api/speakers
```

#### Manually Trigger Model Download (if needed)
```bash
curl -X POST http://localhost:8000/api/download-model
```

### WebSocket API

```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:8000/ws/tts');

ws.onopen = () => {
  ws.send(JSON.stringify({
    text: "Hello from WebSocket!",
    speaker_name: "Carter"
  }));
};

ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Handle audio chunk
    console.log('Received audio chunk');
  } else {
    // Handle JSON status message
    const data = JSON.parse(event.data);
    console.log(data);
  }
};
```

### Python Client Example

```python
import requests
import json

# REST API
url = "http://localhost:8000/api/tts"
data = {
    "text": "Welcome to VibeVoice realtime text to speech!",
    "speaker_name": "Carter",
    "stream": False
}

response = requests.post(url, json=data)
print(response.json())

# WebSocket streaming
import asyncio
import websockets

async def stream_tts():
    uri = "ws://localhost:8000/ws/tts"
    async with websockets.connect(uri) as websocket:
        # Send text
        await websocket.send(json.dumps({
            "text": "Real-time streaming text to speech",
            "speaker_name": "Carter"
        }))
        
        # Receive audio chunks
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                # Process audio chunk
                print(f"Received {len(message)} bytes")
            else:
                # Status message
                data = json.loads(message)
                if data.get("status") == "complete":
                    break

asyncio.run(stream_tts())
```

## Configuration

Edit `.env` file to customize:

- `MODEL_PATH`: Path where model will be downloaded (default: /models/VibeVoice-Realtime-0.5B)
- `MODEL_REPO`: Hugging Face model repository (default: microsoft/VibeVoice-Realtime-0.5B)
- `AUTO_DOWNLOAD_MODEL`: Enable automatic model download (default: true)
- `DOWNLOAD_EXPERIMENTAL_VOICES`: Download additional voices (default: false)
- `PORT`: API port (default: 8000)
- `CUDA_VISIBLE_DEVICES`: GPU device ID
- `HF_TOKEN`: Optional Hugging Face token for private models or better download speeds

### First Run Behavior

On first startup:
1. Container starts and checks for model at `MODEL_PATH`
2. If not found and `AUTO_DOWNLOAD_MODEL=true`, downloads from Hugging Face (~2GB)
3. Model is cached in Docker volume - subsequent starts are instant
4. Download takes 5-10 minutes depending on connection speed

## Production Deployment

### With Nginx Reverse Proxy

```bash
# Start with Nginx
docker-compose --profile production up -d
```

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream vibevoice {
        server vibevoice-api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://vibevoice;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /ws/ {
            proxy_pass http://vibevoice;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Model Download Issues
- Check Docker logs: `docker-compose logs -f`
- Verify internet connection
- Set `HF_TOKEN` in .env if you have rate limits
- Manually trigger download: `curl -X POST http://localhost:8000/api/download-model`
- Check available disk space (need ~5GB free)

### Model Not Loading
- Wait for download to complete (check logs)
- Verify `AUTO_DOWNLOAD_MODEL=true` in .env
- Check model exists: `docker-compose exec vibevoice-api ls -lh /models/VibeVoice-Realtime-0.5B`

### Performance Issues
- Use T4 GPU or better
- Reduce concurrent requests
- Enable flash attention in Dockerfile (uncomment line)
- First request after startup may be slower

## Development

```bash
# Run locally without Docker (for development)
# Install dependencies locally
pip install -r requirements.txt
cd /path/to/VibeVoice
pip install -e .[tts]

# Set environment variables
export AUTO_DOWNLOAD_MODEL=true
export MODEL_PATH=./models/VibeVoice-Realtime-0.5B

# Run the API
python main.py

# Model will auto-download on first run
```

### Disabling Auto-Download

If you want to manually manage models:

```bash
# Set in .env
AUTO_DOWNLOAD_MODEL=false

# Manually download using Hugging Face CLI
pip install huggingface-hub
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir ./models/VibeVoice-Realtime-0.5B
```

## License

This project uses VibeVoice-Realtime by Microsoft. Please review the [VibeVoice license](https://github.com/microsoft/VibeVoice) for usage terms.

## Limitations

- English language only (experimental multilingual support)
- Single speaker in realtime model
- Not recommended for production without further testing
- Research and development purposes

## Contributing

Contributions welcome! Please ensure all changes include:
- Updated documentation
- Docker compatibility testing
- Error handling