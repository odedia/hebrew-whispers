import os
import shutil
import logging
import torch
import torch
import torch
import whisperx
import whisperx
import whisperx.vad 
import whisperx.asr # Import asr to patch load_vad_model reference
import yaml
import traceback
import glob
import pandas as pd
from typing import Optional
from transformers import Wav2Vec2Processor

# MONKEYPATCH: Fix Wav2Vec2Processor.sampling_rate attribute missing in newer transformers
if not hasattr(Wav2Vec2Processor, 'sampling_rate'):
    def get_sampling_rate(self):
        return self.feature_extractor.sampling_rate
    Wav2Vec2Processor.sampling_rate = property(get_sampling_rate)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = os.path.abspath(models_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = None
        self.diarize_model = None
        self.align_model = None
        self.metadata = None

        # Set torch hub dir to local models/torch_hub to avoid runtime download of VAD
        torch_hub_dir = os.path.join(self.models_dir, "torch_hub")
        logger.info(f"Setting torch hub dir to: {torch_hub_dir}")
        os.makedirs(torch_hub_dir, exist_ok=True)
        torch.hub.set_dir(torch_hub_dir)

        logger.info(f"Initializing TranscriptionService on device: {self.device}, compute_type: {self.compute_type}")
        self._load_models()

    def _prepare_offline_diarization_config(self):
        """
        Rewrites the Pyannote config.yaml to point to local model files
        instead of Hugging Face repo IDs.
        """
        diarization_dir = os.path.join(self.models_dir, "pyannote", "diarization")
        config_path = os.path.join(diarization_dir, "config.yaml")

        if not os.path.exists(config_path):
             logger.warning(f"Diarization config not found at {config_path}. Using default online loading.")
             return None

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update paths to point to absolute local paths
            # The structure of downloaded models (via snapshot_download) includes pytorch_model.bin 
            # inside the respective directories.
            
            segmentation_path = os.path.join(self.models_dir, "pyannote", "segmentation", "pytorch_model.bin")
            embedding_path = os.path.join(self.models_dir, "pyannote", "wespeaker", "pytorch_model.bin")
            
            # Check if using segmentation-3.0 repo structure
            if not os.path.exists(segmentation_path):
                 # Try finding it in the folder structure (sometimes snapshots have subfolders)
                 # But we assume flat download to the target dir in download_models.py
                 pass

            # Update config
            # config['pipeline']['params']['segmentation'] = segmentation_path
            # config['pipeline']['params']['embedding'] = embedding_path
            
            # Use os.path.join(self.models_dir, ...) directly in the yaml
            # Pyannote pipeline expects a path to the model checkpoint or a HF ID.
            
            # IMPORTANT: pyannote config.yaml structure:
            # pipeline:
            #   name: pyannote.audio.pipelines.SpeakerDiarization
            #   params:
            #      segmentation: pyannote/segmentation-3.0
            #      embedding: ...
            
            if 'pipeline' in config and 'params' in config['pipeline']:
                 config['pipeline']['params']['segmentation'] = os.path.join(self.models_dir, "pyannote", "segmentation", "pytorch_model.bin")
                 config['pipeline']['params']['embedding'] = os.path.join(self.models_dir, "pyannote", "wespeaker", "pytorch_model.bin")
            
            # Save to a temporary or overwrite
            offline_config_path = os.path.join(diarization_dir, "config_offline.yaml")
            with open(offline_config_path, 'w') as f:
                yaml.dump(config, f)
            
            logger.info(f"Created offline Diarization config at {offline_config_path}")
            return offline_config_path

        except Exception as e:
            logger.error(f"Failed to prepare offline diarization config: {e}")
            return None

    def _load_models(self):
        try:
            logger.info("Loading WhisperX model...")
            whisper_model_path = os.path.join(self.models_dir, "faster-whisper", "large-v3")
            
            if not os.path.exists(whisper_model_path):
                 logger.error(f"Whisper model path not found: {whisper_model_path}")
                 raise FileNotFoundError(f"Whisper model not found at {whisper_model_path}")

            # MONKEYPATCH: Override whisperx.vad.load_vad_model to use local path
            # Find the local VAD directory in models/torch_hub
            vad_root = os.path.join(self.models_dir, "torch_hub")
            # Usually snakers4_silero-vad_master or similiar. Find the first directory that looks like it.
            vad_dirs = glob.glob(os.path.join(vad_root, "snakers4_silero-vad_*"))
            
            if vad_dirs:
                local_vad_path = vad_dirs[0]
                logger.info(f"Found local VAD model at: {local_vad_path}")
                
                # Define custom loader
                def custom_load_vad_model(device, vad_onset=0.500, vad_offset=0.363, use_auth_token=None, **kwargs):
                    logger.info(f"Loading VAD model from local source: {local_vad_path}")
                    # Trust repo is required even for local, checks hubconf.py
                    model, utils = torch.hub.load(repo_or_dir=local_vad_path, model='silero_vad', source='local', onnx=False, trust_repo=True)
                    
                    # Custom wrapper to avoid pyannote/whisperx inheritance issues
                    class OfflineVADWrapper:
                        def __init__(self, model, get_speech_timestamps):
                            self.model = model
                            self.get_speech_timestamps = get_speech_timestamps
                        
                        def __call__(self, audio, **kwargs):
                            # WhisperX passes a dictionary {"waveform": ..., "sample_rate": ...}
                            if isinstance(audio, dict):
                                audio = audio['waveform']

                            # audio can be numpy or tensor. Silero expects Tensor.
                            if not torch.is_tensor(audio):
                                audio = torch.tensor(audio)
                            
                            # Ensure audio is float32
                            if audio.dtype != torch.float32:
                                audio = audio.float()

                            # If 2D (channels, samples), mixdown or take channel 0. 
                            # WhisperX usually passes (samples, ) or (1, samples)
                            if audio.ndim > 1:
                                if audio.shape[0] > 1: 
                                    audio = audio.mean(dim=0, keepdim=True)
                                # Flatten for Silero (expects 1D tensor usually for some calls, or (1, N))
                                # get_speech_timestamps doc says: audio (Tensor): audio signal (1, N) or (N, )
                                if audio.ndim == 2 and audio.shape[0] == 1:
                                    audio = audio.squeeze(0)

                            timestamps = self.get_speech_timestamps(audio, self.model, threshold=0.5)
                            return timestamps

                    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
                    vad_model = OfflineVADWrapper(model, get_speech_timestamps)
                    
                    return vad_model
                
                    return vad_model
                
                # Custom merge_chunks to convert Silero timestamps (samples) to DataFrame (seconds)
                # Bypasses pyannote binarize logic
                def custom_merge_chunks(segments, chunk_size, onset: float = 0.5, offset: float = 0.5):
                    logger.info("Using custom merge_chunks for Silero VAD output")
                    # segments is the list of dicts from Silero: [{'start': 500, 'end': 1000}, ...]
                    
                    if not segments:
                        logger.warning("No speech detected by VAD.")
                        return pd.DataFrame(columns=['start', 'end'])
                        
                    # Convert samples to seconds (assuming 16kHz which Silero/WhisperX uses)
                    sr = 16000
                    data = []
                    for seg in segments:
                        start_sec = seg['start'] / sr
                        end_sec = seg['end'] / sr
                        data.append({'start': start_sec, 'end': end_sec})
                    
                    return data

                # Apply monkeypatch
                whisperx.vad.load_vad_model = custom_load_vad_model
                whisperx.asr.load_vad_model = custom_load_vad_model
                
                # Monkeypatch merge_chunks to handle our custom output
                whisperx.vad.merge_chunks = custom_merge_chunks
                # Also patch asr because it imports merge_chunks
                if hasattr(whisperx.asr, 'merge_chunks'):
                    whisperx.asr.merge_chunks = custom_merge_chunks
                
                logger.info("Monkeypatched whisperx.vad.load_vad_model, asr.load_vad_model, and merge_chunks for offline use.")
            else:
                logger.warning(f"Could not find local VAD model in {vad_root}. WhisperX might try to download it.")

            self.model = whisperx.load_model(
                whisper_model_path, 
                self.device, 
                compute_type=self.compute_type, 
                language="he" 
            )
            logger.info("WhisperX model loaded.")

            # Load Alignment Model
            logger.info("Loading Alignment model...")
            # Use local wav2vec2 model if available
            # Expected path: models/alignment/imvladikon/wav2vec2-xls-r-300m-hebrew
            align_model_path = os.path.join(self.models_dir, "alignment", "imvladikon", "wav2vec2-xls-r-300m-hebrew")
            
            if os.path.exists(align_model_path):
                 logger.info(f"Loading local alignment model from {align_model_path}")
                 self.align_model, self.metadata = whisperx.load_align_model(
                      language_code="he", 
                      device=self.device,
                      model_name=align_model_path
                 )
            else:
                 logger.warning(f"Local alignment model not found at {align_model_path}, attempting default load (may fail air-gapped)...")
                 self.align_model, self.metadata = whisperx.load_align_model(language_code="he", device=self.device)
                 
            logger.info("Alignment model loaded.")

            # Load Diarization Model
            logger.info("Loading Diarization model...")
            
            # Prepare offline config
            offline_config = self._prepare_offline_diarization_config()
            
            if offline_config:
                 logger.info(f"Loading Diarization pipeline from offline config: {offline_config}")
                 self.diarize_model = whisperx.DiarizationPipeline(
                      model_name=offline_config,
                      use_auth_token=os.getenv("HF_TOKEN"), # Still pass if available, but should work without
                      device=self.device
                 )
            else:
                 logger.warning("Offline config preparation failed, falling back to default (online) loading.")
                 self.diarize_model = whisperx.DiarizationPipeline(
                     use_auth_token=os.getenv("HF_TOKEN"),
                     device=self.device
                 )
            logger.info("Diarization model loaded.")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            pass

    def transcribe(self, audio_path: str):
        if not self.model:
            raise RuntimeError("Whisper model not initialized")

        # 1. Transcribe
        logger.info(f"Transcribing {audio_path}...")
        result = self.model.transcribe(audio_path, batch_size=16)
        
        # 2. Align (Optional but requested)
        if self.align_model:
            logger.info("Aligning transcription...")
            try:
                result = whisperx.align(result["segments"], self.align_model, self.metadata, audio_path, self.device, return_char_alignments=False)
            except Exception as e:
                logger.error(f"Alignment failed: {e}")

        # 3. Diarize
        if self.diarize_model:
            logger.info("Diarizing speakers...")
            try:
                diarize_segments = self.diarize_model(audio_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                 logger.error(f"Diarization failed: {e}")

        return result
