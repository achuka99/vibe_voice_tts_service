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
    
    # Fix for torch.xpu error - monkey patch if it doesn't exist
    if not hasattr(torch, 'xpu'):
        logger.info("Adding torch.xpu compatibility layer")
        class XPUDevice:
            def __init__(self):
                self.is_available = lambda: False
                self.device_count = lambda: 0
                self.current_device = lambda: 0
                self.empty_cache = lambda: None
                self.synchronize = lambda: None
                self.manual_seed = lambda seed: None
                self.initial_seed = lambda: 0
                self.get_rng_state = lambda: None
                self.set_rng_state = lambda state: None
        
        torch.xpu = XPUDevice()
    
    try:
        # Import the inference model and processor
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        
        logger.info(f"Loading VibeVoice model from {model_path} on device {device}")
        
        # Set environment variables as done in the demo
        os.environ["MODEL_PATH"] = model_path
        os.environ["MODEL_DEVICE"] = device
        
        # Load the processor
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
        
        # Create the model instance
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        
        # Create a wrapper that provides the expected interface
        class VibeVoiceModelWrapper:
            def __init__(self, model, processor):
                self.model = model
                self.processor = processor
            
            def generate(self, text: str, speaker_name: str = "Carter"):
                """Generate audio from text"""
                # Process the text
                inputs = self.processor(text=text, return_tensors="pt")
                
                # Move to model device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.model.device)
                
                # Generate audio
                with torch.no_grad():
                    outputs = self.model.generate(**inputs)
                
                # Return audio data (convert to numpy if needed)
                return outputs.audio_values.cpu().numpy()
            
            def generate_streaming(self, text: str, speaker_name: str = "Carter"):
                """Generate streaming audio from text"""
                # For now, just return the full generation in chunks
                audio = self.generate(text, speaker_name)
                
                # Split into chunks
                chunk_size = 1024
                audio_bytes = audio.tobytes()
                
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i+chunk_size]
        
        wrapped_model = VibeVoiceModelWrapper(model, processor)
        
        logger.info("✅ VibeVoice model loaded successfully!")
        return wrapped_model
        
    except ImportError as e:
        logger.error(f"Failed to import VibeVoice classes: {e}")
        raise ImportError("Could not find VibeVoice model class. Check the installation.")
        
    except Exception as e:
        logger.error(f"Failed to load VibeVoice model: {e}")
        raise e
