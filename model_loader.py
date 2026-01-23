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
from typing import Optional, Iterator, Callable, Dict, Any
import threading

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
            
            def _get_voice_resources(self, voice_key: Optional[str] = None):
                """Get voice resources like official demo"""
                import glob
                
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
                voice_name_key = speaker_mapping.get(voice_key, "en-Carter_man")
                voice_path = f"/app/vibevoice/demo/voices/streaming_model/{voice_name_key}.pt"
                
                # First try exact match for speaker
                voice_files = glob.glob(f"/app/vibevoice/demo/voices/streaming_model/{voice_name_key}*.pt")
                if voice_files:
                    voice_path = voice_files[0]
                else:
                    # Try to find any English voice file
                    voice_files = glob.glob("/app/vibevoice/demo/voices/streaming_model/en-*.pt")
                    if voice_files:
                        voice_path = voice_files[0]
                    else:
                        # Fallback to any voice file
                        voice_files = glob.glob("/app/vibevoice/demo/voices/streaming_model/*.pt")
                        if voice_files:
                            voice_path = voice_files[0]
                        else:
                            raise FileNotFoundError("No voice files found")
                
                logger.info(f"Loading voice preset from: {voice_path}")
                cached_prompt = torch.load(voice_path, map_location=self.model.device, weights_only=False)
                
                return voice_path, cached_prompt

            def _prepare_inputs(self, text: str, prefilled_outputs):
                """Prepare inputs like official demo"""
                inputs = self.processor.process_input_with_cached_prompt(
                    text=text,
                    cached_prompt=prefilled_outputs,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True
                )
                
                # Move to model device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.model.device)
                
                return inputs

            def _run_generation(
                self,
                inputs,
                audio_streamer: AudioStreamer,
                errors,
                cfg_scale: float,
                do_sample: bool,
                temperature: float,
                top_p: float,
                refresh_negative: bool,
                prefilled_outputs,
                stop_event,
            ):
                """Run generation like official demo"""
                try:
                    logger.info(f"Starting generation with params: do_sample={do_sample}, temperature={temperature}, top_p={top_p}, cfg_scale={cfg_scale}")
                    
                    # Use the exact same parameters as official demo
                    self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={
                            "do_sample": do_sample,
                            "temperature": temperature if do_sample else 1.0,
                            "top_p": top_p if do_sample else 1.0,
                        },
                        audio_streamer=audio_streamer,  # ✅ CRITICAL - Connect AudioStreamer!
                        stop_check_fn=stop_event.is_set,
                        verbose=False,
                        refresh_negative=refresh_negative,
                        all_prefilled_outputs=copy.deepcopy(prefilled_outputs),  # ✅ CRITICAL - Fix NoneType error
                    )
                    
                    logger.info("Generation completed successfully")
                except Exception as exc:
                    logger.error(f"Generation error: {exc}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    errors.append(exc)
                    audio_streamer.end()

            def generate_streaming(
                self, 
                text: str, 
                cfg_scale: float = 1.5,
                do_sample: bool = True,  # ✅ Changed to True like official demo
                temperature: float = 0.9,
                top_p: float = 0.9,
                refresh_negative: bool = True,
                inference_steps: Optional[int] = None,
                voice_key: Optional[str] = None,
                log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                stop_event: Optional[threading.Event] = None,
            ) -> Iterator[np.ndarray]:
                """Generate streaming audio from text using AudioStreamer - exactly like official demo"""
                try:
                    if not text.strip():
                        return
                    text = text.replace("'", "'")
                    
                    logger.info(f"Starting streaming generation for: {text[:50]}...")
                    logger.info(f"Parameters: CFG={cfg_scale}, Steps={inference_steps}, do_sample={do_sample}, temperature={temperature}, top_p={top_p}")
                    
                    # Use voice_key like official demo
                    selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

                    def emit(event: str, **payload: Any) -> None:
                        if log_callback:
                            try:
                                log_callback(event, **payload)
                            except Exception as exc:
                                print(f"[log_callback] Error while emitting {event}: {exc}")

                    steps_to_use = 5  # Default inference steps
                    if inference_steps is not None:
                        try:
                            parsed_steps = int(inference_steps)
                            if parsed_steps > 0:
                                steps_to_use = parsed_steps
                        except (TypeError, ValueError):
                            pass
                    
                    if self.model:
                        self.model.set_ddpm_inference_steps(num_steps=steps_to_use)

                    inputs = self._prepare_inputs(text, prefilled_outputs)
                    logger.info(f"Inputs prepared: {list(inputs.keys())}")
                    
                    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
                    errors: list = []
                    stop_signal = stop_event or threading.Event()

                    logger.info("Starting generation thread...")
                    thread = threading.Thread(
                        target=self._run_generation,
                        kwargs={
                            "inputs": inputs,
                            "audio_streamer": audio_streamer,
                            "errors": errors,
                            "cfg_scale": cfg_scale,
                            "do_sample": do_sample,
                            "temperature": temperature,
                            "top_p": top_p,
                            "refresh_negative": refresh_negative,
                            "prefilled_outputs": prefilled_outputs,
                            "stop_event": stop_signal,
                        },
                        daemon=True,
                    )
                    thread.start()

                    generated_samples = 0
                    logger.info("Starting to receive audio stream...")

                    try:
                        stream = audio_streamer.get_stream(0)
                        chunk_count = 0
                        logger.info("Starting to iterate over audio stream...")
                        
                        for audio_chunk in stream:
                            chunk_count += 1
                            logger.info(f"Received chunk {chunk_count}: {type(audio_chunk)}, shape: {getattr(audio_chunk, 'shape', 'N/A')}")
                            
                            if torch.is_tensor(audio_chunk):
                                audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                            else:
                                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                            if audio_chunk.ndim > 1:
                                audio_chunk = audio_chunk.reshape(-1)

                            peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                            if peak > 1.0:
                                audio_chunk = audio_chunk / peak

                            generated_samples += int(audio_chunk.size)
                            emit(
                                "model_progress",
                                generated_sec=generated_samples / 24000,
                                chunk_sec=audio_chunk.size / 24000,
                            )

                            chunk_to_yield = audio_chunk.astype(np.float32, copy=False)
                            logger.info(f"Yielding chunk {chunk_count}: {chunk_to_yield.nbytes} bytes")
                            yield chunk_to_yield
                        
                        logger.info(f"Stream completed after {chunk_count} chunks")
                        
                        # Add debug info about the stream
                        if chunk_count == 0:
                            logger.warning("NO CHUNKS WERE YIELDED FROM AUDIO STREAMER!")
                            logger.warning("This suggests AudioStreamer is not producing audio chunks properly")
                        
                    finally:
                        logger.info("Cleaning up...")
                        stop_signal.set()
                        audio_streamer.end()
                        thread.join()
                        if errors:
                            emit("generation_error", message=str(errors[0]))
                            raise errors[0]
                        
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
