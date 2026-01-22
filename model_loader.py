"""
Model loader for VibeVoice based on the official documentation
"""

# Apply XPU compatibility patch BEFORE any other imports
import torch
if not hasattr(torch, 'xpu'):
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
        # Import the inference model and processor
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        
        logger.info(f"Loading VibeVoice model from {model_path} on device {device}")
        
        # Set environment variables as done in the demo
        os.environ["MODEL_PATH"] = model_path
        os.environ["MODEL_DEVICE"] = device
        
        # Decide dtype & attention implementation based on device
        if device == "mps":
            load_dtype = torch.float32  # MPS requires float32
            attn_impl_primary = "sdpa"  # flash_attention_2 not supported on MPS
        elif device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = "flash_attention_2"
        else:  # cpu
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        
        logger.info(f"Using device: {device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        
        # Load the processor
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
        logger.info(f"Processor loaded successfully")
        logger.info(f"Processor attributes: {[attr for attr in dir(processor) if not attr.startswith('_')]}")
        
        # Check if tokenizer exists
        if hasattr(processor, 'tokenizer'):
            logger.info(f"Tokenizer found: {type(processor.tokenizer)}")
        else:
            logger.warning("No tokenizer attribute found in processor")
        
        # Try to access text tokenizer
        if hasattr(processor, 'text_tokenizer'):
            logger.info(f"Text tokenizer found: {type(processor.text_tokenizer)}")
        else:
            logger.warning("No text_tokenizer attribute found in processor")
        
        # Load model with device-specific logic
        try:
            if device == "mps":
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,  # load then move
                )
                model = model.to(device)
            else:
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=device if device == "cuda" else None,
                )
        except Exception as e:
            logger.warning(f"Failed to load with flash attention, falling back to SDPA: {e}")
            # Fallback to SDPA if flash attention fails
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                attn_implementation="sdpa",
                device_map=device if device == "cuda" else None,
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
                try:
                    # Load voice preset for the speaker
                    voice_path = f"/app/vibevoice/demo/voices/streaming_model/en-{speaker_name}_man.pt"
                    import glob
                    # First try exact match for the speaker
                    voice_files = glob.glob(f"/app/vibevoice/demo/voices/streaming_model/en-{speaker_name}_*.pt")
                    if voice_files:
                        voice_path = voice_files[0]
                    else:
                        # Try to find any English voice file
                        voice_files = glob.glob("/app/vibevoice/demo/voices/streaming_model/en-*.pt")
                        if voice_files:
                            voice_path = voice_files[0]
                            logger.warning(f"Speaker '{speaker_name}' not found, using default voice: {voice_path}")
                        else:
                            # Fallback to any voice file
                            voice_files = glob.glob("/app/vibevoice/demo/voices/streaming_model/*.pt")
                            if voice_files:
                                voice_path = voice_files[0]
                                logger.warning(f"No English voice found, using default voice: {voice_path}")
                            else:
                                raise FileNotFoundError("No voice files found")
                    
                    logger.info(f"Loading voice preset from: {voice_path}")
                    cached_prompt = torch.load(voice_path, map_location=self.model.device, weights_only=False)
                    
                    # Debug processor before processing
                    logger.info(f"Processor type: {type(self.processor)}")
                    if hasattr(self.processor, 'text_tokenizer'):
                        logger.info(f"Text tokenizer type: {type(self.processor.text_tokenizer)}")
                        if self.processor.text_tokenizer is None:
                            logger.error("Text tokenizer is None!")
                    
                    # Process the text using the correct method
                    inputs = self.processor.process_input_with_cached_prompt(
                        text=text,
                        cached_prompt=cached_prompt,
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True
                    )
                    
                    # Move to model device
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(self.model.device)
                    
                    # Generate audio
                    with torch.no_grad():
                        # Pass the tokenizer from the processor to the model's generate method
                        # Also pass the cached_prompt as all_prefilled_outputs
                        outputs = self.model.generate(
                            **inputs,
                            tokenizer=getattr(self.processor, 'text_tokenizer', None) or getattr(self.processor, 'tokenizer', None),
                            all_prefilled_outputs=cached_prompt
                        )
                    
                    # Return audio data (convert to numpy if needed)
                    return outputs.audio_values.cpu().numpy()
                    
                except Exception as e:
                    logger.error(f"Error in generate method: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise e
            
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
