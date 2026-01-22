"""
Model loader for VibeVoice based on the Colab notebook implementation
"""
import torch
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

def load_vibevoice_model(model_path: str, device: str = "cpu"):
    """
    Load VibeVoice model using the actual API from the repository
    """
    # Add VibeVoice to path
    sys.path.insert(0, "/app/vibevoice")
    
    # Import the demo module which contains the model loading logic
    from demo.vibevoice_realtime_demo import VibeVoiceRealtimeDemo
    
    logger.info(f"Loading VibeVoice model from {model_path} on device {device}")
    
    # Create the demo instance which loads the model
    model = VibeVoiceRealtimeDemo(model_path=model_path, device=device)
    
    logger.info("✅ VibeVoice model loaded successfully!")
    return model
