# Use NVIDIA PyTorch base image for CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone VibeVoice repository
RUN git clone https://github.com/microsoft/VibeVoice.git /app/vibevoice

# Install VibeVoice dependencies first
WORKDIR /app/vibevoice
RUN pip install --no-cache-dir -e .[tts]

# Install FastAPI and related dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==12.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1 \
    python-dotenv==1.0.0 \
    huggingface-hub \
    hf_transfer

# Optional: Install flash attention for better performance
# Uncomment if needed and if your GPU supports it
# RUN pip install flash-attn --no-build-isolation

# Copy FastAPI application
WORKDIR /app
COPY main.py .

# Create directories for models and audio
# Models will be auto-downloaded on first run
RUN mkdir -p /models /output

# Expose port
EXPOSE 8000

# Environment variables
ENV MODEL_PATH=/models/VibeVoice-Realtime-0.5B
ENV MODEL_REPO=microsoft/VibeVoice-Realtime-0.5B
ENV AUTO_DOWNLOAD_MODEL=true
ENV DOWNLOAD_EXPERIMENTAL_VOICES=false
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]