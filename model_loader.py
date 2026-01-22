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
    
    # Try to find the correct import by checking what's available
    try:
        # First, let's see what's in the demo module
        import demo.vibevoice_realtime_demo as demo_module
        logger.info(f"Available attributes in demo module: {dir(demo_module)}")
        
        # Look for the main class or function
        for attr_name in dir(demo_module):
            if 'VibeVoice' in attr_name or 'Demo' in attr_name:
                logger.info(f"Found potential class: {attr_name}")
        
        # Try to import the main class (common patterns)
        if hasattr(demo_module, 'VibeVoiceRealtimeDemo'):
            model_class = demo_module.VibeVoiceRealtimeDemo
        elif hasattr(demo_module, 'VibeVoiceRealtime'):
            model_class = demo_module.VibeVoiceRealtime
        elif hasattr(demo_module, 'VibeVoice'):
            model_class = demo_module.VibeVoice
        else:
            # Try to import from the main vibevoice module
            import vibevoice
            logger.info(f"Available in vibevoice module: {dir(vibevoice)}")
            if hasattr(vibevoice, 'VibeVoiceRealtime'):
                model_class = vibevoice.VibeVoiceRealtime
            else:
                raise ImportError("Could not find VibeVoice model class")
        
        logger.info(f"Loading VibeVoice model from {model_path} on device {device}")
        
        # Create the model instance
        model = model_class(model_path=model_path, device=device)
        
        logger.info("✅ VibeVoice model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load VibeVoice model: {e}")
        raise e
