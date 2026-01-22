"""
Model loader for VibeVoice based on the official documentation
"""
import torch
import logging
from pathlib import Path
import sys
import os

logger = logging.getLogger(__name__)

def load_vibevoice_model(model_path: str, device: str = "cpu"):
    """
    Load VibeVoice model using the actual API from the repository
    """
    # Add VibeVoice to path
    sys.path.insert(0, "/app/vibevoice")
    
    try:
        # Import the actual model from the VibeVoice package
        # Based on the documentation, we need to import the model class
        from vibevoice.models.vibevoice_realtime import VibeVoiceRealtime
        
        logger.info(f"Loading VibeVoice model from {model_path} on device {device}")
        
        # Set environment variables as done in the demo
        os.environ["MODEL_PATH"] = model_path
        os.environ["MODEL_DEVICE"] = device
        
        # Create the model instance
        model = VibeVoiceRealtime(model_path=model_path, device=device)
        
        logger.info("✅ VibeVoice model loaded successfully!")
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import VibeVoiceRealtime: {e}")
        # Try alternative import paths
        try:
            from vibevoice import VibeVoiceRealtime
            logger.info("Using alternative import path")
            
            os.environ["MODEL_PATH"] = model_path
            os.environ["MODEL_DEVICE"] = device
            
            model = VibeVoiceRealtime(model_path=model_path, device=device)
            logger.info("✅ VibeVoice model loaded successfully!")
            return model
        except ImportError as e2:
            logger.error(f"Alternative import also failed: {e2}")
            
            # Try to find what's available in the vibevoice package
            try:
                import vibevoice
                logger.info(f"Available in vibevoice package: {[x for x in dir(vibevoice) if not x.startswith('_')]}")
                
                # Look for model-related modules
                vibevoice_path = Path("/app/vibevoice")
                models_path = vibevoice_path / "vibevoice" / "models"
                if models_path.exists():
                    logger.info(f"Available models: {[f.name for f in models_path.iterdir() if f.is_file() and f.name.endswith('.py')]}")
                
            except Exception as e3:
                logger.error(f"Could not explore vibevoice package: {e3}")
            
            raise ImportError("Could not find VibeVoice model class. Check the installation.")
        
    except Exception as e:
        logger.error(f"Failed to load VibeVoice model: {e}")
        raise e
