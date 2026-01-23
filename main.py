"""
VibeVoice-Realtime FastAPI Backend
Production-ready API for real-time text-to-speech with automatic model downloading
"""
import logging
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import asyncio
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_vibevoice_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VibeVoice-Realtime API",
    description="Real-time text-to-speech with streaming capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "/models/VibeVoice-Realtime-0.5B")
MODEL_REPO = os.getenv("MODEL_REPO", "microsoft/VibeVoice-Realtime-0.5B")
AUTO_DOWNLOAD = os.getenv("AUTO_DOWNLOAD_MODEL", "true").lower() == "true"

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    speaker_name: str = Field(default="Carter", description="Speaker voice name")
    stream: bool = Field(default=False, description="Enable streaming response")
    max_new_tokens: Optional[int] = Field(default=8192, description="Maximum number of new tokens to generate")
    temperature: Optional[float] = Field(default=1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling")

class TTSResponse(BaseModel):
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    message: str

async def download_model():
    """Download model from Hugging Face if not present"""
    model_path = Path(MODEL_PATH)
    
    # Check if model actually exists with model files (not just cache)
    if model_path.exists():
        # Look for actual model files, not just cache
        model_files = list(model_path.glob("*.json")) + list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        if model_files:
            logger.info(f"Model already exists at {MODEL_PATH} with {len(model_files)} files")
            return True
        else:
            logger.info(f"Model directory exists but no model files found at {MODEL_PATH}")
    
    if not AUTO_DOWNLOAD:
        logger.error(f"Model not found at {MODEL_PATH} and AUTO_DOWNLOAD is disabled")
        return False
    
    try:
        logger.info(f"Downloading model {MODEL_REPO} to {MODEL_PATH}")
        logger.info("This may take several minutes on first run...")
        
        # Remove existing directory if it's incomplete (only cache)
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download model from Hugging Face
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        # Verify download was successful
        model_files = list(model_path.glob("*.json")) + list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        if model_files:
            logger.info(f"Model downloaded successfully! Found {len(model_files)} model files")
            return True
        else:
            logger.error("Download completed but no model files found")
            return False
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

async def download_experimental_voices():
    """Download experimental voices if enabled"""
    if not os.getenv("DOWNLOAD_EXPERIMENTAL_VOICES", "false").lower() == "true":
        return
    
    try:
        logger.info("Downloading experimental voices...")
        experimental_path = Path("/models/experimental_voices")
        experimental_path.mkdir(parents=True, exist_ok=True)
        
        # Download experimental voices
        snapshot_download(
            repo_id="microsoft/VibeVoice-Experimental-Voices",
            local_dir=str(experimental_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info("Experimental voices downloaded!")
    except Exception as e:
        logger.warning(f"Could not download experimental voices: {e}")

async def load_model():
    """Load VibeVoice model on startup"""
    global model
    
    # First, ensure model is downloaded
    if not await download_model():
        raise RuntimeError("Failed to download or locate model")
    
    # Download experimental voices if configured
    await download_experimental_voices()
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = load_vibevoice_model(MODEL_PATH, device)
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting VibeVoice API...")
    logger.info(f"Model repository: {MODEL_REPO}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Auto download: {AUTO_DOWNLOAD}")
    
    try:
        await load_model()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.warning("API will start but TTS endpoints may not work until model is loaded")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": MODEL_REPO,
        "model_path": MODEL_PATH,
        "model_loaded": model is not None,
        "endpoints": {
            "tts": "/api/tts",
            "websocket": "/ws/tts",
            "health": "/health",
            "speakers": "/api/speakers"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_exists = Path(MODEL_PATH).exists()
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_exists": model_exists,
        "model_path": MODEL_PATH,
        "auto_download": AUTO_DOWNLOAD
    }

@app.post("/api/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech
    
    - **text**: Input text to synthesize
    - **speaker_name**: Voice to use (default: Carter)
    - **stream**: Enable streaming response
    - **max_new_tokens**: Maximum number of new tokens to generate
    - **temperature**: Sampling temperature
    - **top_p**: Top-p sampling
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Try restarting the API."
        )
    
    try:
        logger.info(f"Processing TTS request: {len(request.text)} characters")
        
        # Clean up the text to remove invalid characters
        import re
        cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', request.text)
        
        if request.stream:
            # Return streaming audio response
            async def generate_audio() -> AsyncGenerator[bytes, None]:
                audio_chunks = model.generate_streaming(
                    text=cleaned_text,
                    speaker_name=request.speaker_name
                )
                for chunk in audio_chunks:
                    yield chunk
            
            return StreamingResponse(
                generate_audio(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="speech.wav"'
                }
            )
        else:
            # Generate complete audio
            result = model.generate(
                text=cleaned_text,
                speaker_name=request.speaker_name
            )
            
            # Extract audio information from the result
            if isinstance(result, dict):
                audio_data = result.get('audio_data')
                audio_file = result.get('audio_file')
                duration = result.get('duration', 0)
            else:
                # Backward compatibility for direct audio data
                audio_data = result
                audio_file = None
                duration = len(result) / 24000 if hasattr(result, '__len__') else 0
            
            # Return audio file directly as downloadable response
            if audio_data is not None:
                import io
                import wave
                
                # Create WAV file in memory
                wav_buffer = io.BytesIO()
                
                # Convert audio to int16 format
                if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                # Write WAV to buffer
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(audio_int16.tobytes())
                
                wav_buffer.seek(0)
                
                return StreamingResponse(
                    io.BytesIO(wav_buffer.read()),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f'attachment; filename="vibevoice_speech.wav"',
                        "Content-Length": str(len(wav_buffer.getvalue()))
                    }
                )
            else:
                return TTSResponse(
                    audio_path=audio_file,
                    duration=duration,
                    message="Audio generated successfully"
                )
            
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming TTS
    
    Send JSON: {"text": "your text", "speaker_name": "Carter"}
    Receive: Binary audio chunks
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Set binary type for WebSocket
    logger.info("Setting WebSocket binary type to arraybuffer")
    
    if model is None:
        await websocket.send_json({
            "error": "Model not loaded",
            "message": "Use /api/download-model to download model first"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive text input
            data = await websocket.receive_json()
            text = data.get("text", "")
            speaker_name = data.get("speaker_name", "Carter")
            
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue
            
            logger.info(f"WS: Processing {len(text)} characters")
            
            # Generate and stream audio chunks
            try:
                logger.info("Starting audio streaming...")
                
                # Extract inference parameters
                cfg_scale = data.get("cfg_scale", 1.5)
                inference_steps = data.get("inference_steps", 5)
                
                logger.info(f"Inference params: CFG={cfg_scale}, Steps={inference_steps}")
                
                # Use our working generate_streaming method with official demo parameters
                audio_chunks = model.generate_streaming(
                    text=text,
                    voice_key=speaker_name,  # Use voice_key parameter like official demo
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps
                )
                
                chunk_count = 0
                for chunk in audio_chunks:
                    # Convert to PCM16 exactly like official demo
                    chunk = np.clip(chunk, -1.0, 1.0)
                    pcm = (chunk * 32767.0).astype(np.int16)
                    payload = pcm.tobytes()
                    
                    logger.info(f"Sending chunk {chunk_count + 1}: {len(payload)} bytes")
                    await websocket.send_bytes(payload)
                    chunk_count += 1
                
                logger.info(f"Sent {chunk_count} audio chunks")
                # Send completion message
                await websocket.send_json({"status": "complete"})
                
            except Exception as e:
                logger.error(f"WS generation error: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/api/speakers")
async def list_speakers():
    """List available speaker voices"""
    import glob
    
    # Get actual voice files from the container
    voice_files = glob.glob("/app/vibevoice/demo/voices/streaming_model/*.pt")
    
    # Extract speaker names from filenames
    available_speakers = []
    for voice_file in voice_files:
        filename = os.path.basename(voice_file)
        # Remove .pt extension and get the speaker part
        speaker_name = filename.replace('.pt', '')
        # For English voices, extract the actual name (e.g., "en-Carter_man" -> "Carter")
        if speaker_name.startswith('en-'):
            parts = speaker_name.split('-')
            if len(parts) >= 2:
                name_part = parts[1]  # e.g., "Carter_man"
                speaker_name = name_part.split('_')[0]  # e.g., "Carter"
        available_speakers.append(speaker_name)
    
    # Get experimental voices if available
    experimental_path = Path("/models/experimental_voices")
    experimental_speakers = []
    
    if experimental_path.exists():
        experimental_speakers = [
            f.stem for f in experimental_path.glob("*.pt")
        ]
    
    return {
        "available": available_speakers,
        "experimental": experimental_speakers,
        "total_count": len(available_speakers)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
