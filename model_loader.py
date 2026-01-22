"""
Model loader for VibeVoice based on the official documentation
"""

import sys
import os
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

# Apply torch.xpu compatibility fix at the very top
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

logger = logging.getLogger(__name__)

# Add VibeVoice to path
sys.path.insert(0, "/app/vibevoice")

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from vibevoice.modular.streamer import AudioStreamer

def save_audio_as_wav(audio_data, sample_rate=24000, output_dir="/output"):
    """Save audio data as WAV file"""
    try:
        import scipy.io.wavfile as wavfile
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"vibevoice_{timestamp}_{unique_id}.wav"
        filepath = output_path / filename
        
        # Convert audio data to int16 format (16-bit PCM)
        if audio_data.dtype != np.int16:
            # Normalize to [-1, 1] range if needed
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Ensure values are in valid range [-1, 1] before conversion
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Convert to int16
            audio_data_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_data_int16 = audio_data
        
        # Save as WAV file
        wavfile.write(str(filepath), sample_rate, audio_data_int16)
        
        logger.info(f"Audio saved to: {filepath}")
        return str(filepath)
        
    except ImportError:
        logger.warning("scipy not available, trying to save as raw audio")
        # Fallback: save as raw numpy array
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"vibevoice_{timestamp}_{unique_id}.npy"
        filepath = output_path / filename
        
        np.save(str(filepath), audio_data)
        logger.info(f"Audio saved as numpy array to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        return None

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
                    # Map speaker names to voice files
                    speaker_mapping = {
                        "Carter": "en-Carter_man",
                        "Davis": "en-Davis_man", 
                        "Emma": "en-Emma_woman",
                        "Frank": "en-Frank_man",
                        "Grace": "en-Grace_woman",
                        "Mike": "en-Mike_man"
                    }
                    
                    # Get the correct voice file name
                    voice_name_key = speaker_mapping.get(speaker_name, "en-Carter_man")
                    voice_path = f"/app/vibevoice/demo/voices/streaming_model/{voice_name_key}.pt"
                    
                    import glob
                    # First try exact match for speaker
                    voice_files = glob.glob(f"/app/vibevoice/demo/voices/streaming_model/{voice_name_key}*.pt")
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
                    
                    # Debug the output object
                    logger.info(f"Output type: {type(outputs)}")
                    logger.info(f"Output attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                    
                    # Return audio data (convert to numpy if needed)
                    if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs:
                        # Get the first speech output
                        audio_data = outputs.speech_outputs[0]
                        if isinstance(audio_data, torch.Tensor):
                            # Convert to float32 before converting to numpy
                            if audio_data.dtype == torch.bfloat16:
                                audio_data = audio_data.float()
                            audio_numpy = audio_data.cpu().numpy()
                        else:
                            audio_numpy = audio_data
                    elif hasattr(outputs, 'audio_values'):
                        audio_data = outputs.audio_values
                        if audio_data.dtype == torch.bfloat16:
                            audio_data = audio_data.float()
                        audio_numpy = audio_data.cpu().numpy()
                    else:
                        logger.error(f"No audio found in output. Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                        raise AttributeError("No audio data found in model output")
                    
                    # Save audio as WAV file
                    audio_file_path = save_audio_as_wav(audio_numpy)
                    
                    # Return both the audio data and the file path
                    return {
                        'audio_data': audio_numpy,
                        'audio_file': audio_file_path,
                        'sample_rate': 24000,
                        'duration': len(audio_numpy) / 24000
                    }
                    
                except Exception as e:
                    logger.error(f"Error in generate method: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise e
            
            def generate_streaming(self, text: str, speaker_name: str = "Carter", cfg_scale: float = 1.5, inference_steps: int = 5):
                """Generate streaming audio from text using AudioStreamer"""
                try:
                    logger.info(f"Starting streaming generation for: {text[:50]}...")
                    logger.info(f"Parameters: CFG={cfg_scale}, Steps={inference_steps}")
                    
                    # Map speaker names to voice files
                    speaker_mapping = {
                        "Carter": "en-Carter_man",
                        "Davis": "en-Davis_man", 
                        "Emma": "en-Emma_woman",
                        "Frank": "en-Frank_man",
                        "Grace": "en-Grace_woman",
                        "Mike": "en-Mike_man"
                    }
                    
                    # Get the correct voice file name
                    voice_name_key = speaker_mapping.get(speaker_name, "en-Carter_man")
                    voice_path = f"/app/vibevoice/demo/voices/streaming_model/{voice_name_key}.pt"
                    
                    import glob
                    # First try exact match for speaker
                    voice_files = glob.glob(f"/app/vibevoice/demo/voices/streaming_model/{voice_name_key}*.pt")
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
                    
                    # Create AudioStreamer for real-time streaming
                    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
                    logger.info("AudioStreamer created, starting generation...")
                    
                    # Create stop event like official demo
                    stop_event = threading.Event()
                    
                    # Start generation in a separate thread
                    import threading
                    import copy
                    import time
                    
                    def run_generation():
                        try:
                            logger.info("Starting model.generate() in thread...")
                            start_time = time.time()
                            
                            # CRITICAL: Configure noise scheduler exactly like official demo
                            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                                self.model.model.noise_scheduler.config,
                                algorithm_type="sde-dpmsolver++",
                                beta_schedule="squaredcos_cap_v2",
                            )
                            logger.info("Noise scheduler configured with sde-dpmsolver++")
                            
                            # Set inference steps RIGHT before generation (like official demo)
                            self.model.set_ddpm_inference_steps(num_steps=inference_steps)
                            
                            self.model.generate(
                                **inputs,
                                max_new_tokens=None,  # Exactly like official demo
                                cfg_scale=cfg_scale,
                                tokenizer=self.processor.tokenizer,  # Exact tokenizer reference like official demo
                                generation_config={
                                    "do_sample": True,  # Official demo uses do_sample parameter
                                    "temperature": 1.0,  # Fixed like official demo
                                    "top_p": 0.9,  # Fixed like official demo
                                },
                                audio_streamer=audio_streamer,
                                stop_check_fn=stop_event.is_set,  # CRITICAL: Missing piece from official demo!
                                verbose=False,
                                refresh_negative=True,
                                all_prefilled_outputs=copy.deepcopy(cached_prompt),  # Exact like official demo
                            )
                            end_time = time.time()
                            logger.info(f"Model generation completed in {end_time - start_time:.2f}s")
                        except Exception as e:
                            logger.error(f"Generation error: {e}")
                            audio_streamer.end()
                    
                    thread = threading.Thread(target=run_generation, daemon=True)
                    thread.start()
                    
                    chunks_sent = 0
                    # Stream audio chunks as they're generated - use the same approach as official demo
                    try:
                        logger.info("Getting audio stream...")
                        stream = audio_streamer.get_stream(0)
                        
                        # Use the official demo approach - simple for loop over stream
                        for audio_chunk in stream:
                            logger.info(f"Received audio chunk: {type(audio_chunk)}, shape: {getattr(audio_chunk, 'shape', 'N/A')}")
                            
                            if torch.is_tensor(audio_chunk):
                                audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                            else:
                                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
                            
                            if audio_chunk.ndim > 1:
                                audio_chunk = audio_chunk.reshape(-1)
                            
                            logger.info(f"Audio chunk shape after processing: {audio_chunk.shape}")
                            
                            # Normalize and convert to PCM16
                            peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                            if peak > 1.0:
                                audio_chunk = audio_chunk / peak
                            
                            # Clip to [-1, 1] and convert to int16
                            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                            pcm16 = (audio_chunk * 32767.0).astype(np.int16)
                            
                            logger.info(f"Yielding chunk {chunks_sent + 1}: {pcm16.nbytes} bytes")
                            chunks_sent += 1
                            
                            # Yield as bytes
                            yield pcm16.tobytes()
                        
                        # Wait for generation thread to complete
                        thread.join(timeout=30)
                        logger.info(f"Stream completed naturally, sent {chunks_sent} chunks total")
                        
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        raise e
                    finally:
                        logger.info(f"Stream ended, sent {chunks_sent} chunks")
                        audio_streamer.end()
                        if thread.is_alive():
                            logger.warning("Generation thread still running after stream end")
                            thread.join(timeout=5)
                        
                except Exception as e:
                    logger.error(f"Error in generate_streaming method: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise e
        
        wrapped_model = VibeVoiceModelWrapper(model, processor)
        
        # CRITICAL: Put model in eval mode like official demo
        model.eval()
        logger.info("Model set to eval mode")
        
        logger.info("✅ VibeVoice model loaded successfully!")
        return wrapped_model
        
    except ImportError as e:
        logger.error(f"Failed to import VibeVoice classes: {e}")
        raise ImportError("Could not find VibeVoice model class. Check the installation.")
        
    except Exception as e:
        logger.error(f"Failed to load VibeVoice model: {e}")
        raise e
