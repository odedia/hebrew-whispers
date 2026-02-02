import argparse
import os
import shutil
import logging
import sys
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
import torch
import whisperx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
HF_TOKEN = os.getenv("HF_TOKEN")

def download_faster_whisper(model_size="large-v3"):
    output_dir = os.path.join(MODELS_DIR, "faster-whisper", model_size)
    model_bin = os.path.join(output_dir, "model.bin")
    
    # Check if model.bin exists and is not empty (arbitrary check > 1KB)
    if os.path.exists(model_bin) and os.path.getsize(model_bin) > 1024:
        logger.info(f"Faster Whisper model {model_size} verified at {output_dir}. Skipping.")
        return

    if os.path.exists(output_dir):
        logger.warning(f"Found existing directory {output_dir} but model.bin invalid/missing. Cleaning up.")
        shutil.rmtree(output_dir)

    logger.info(f"Downloading Faster Whisper model: {model_size}...")
    try:
        # download to a specific directory within MODELS_DIR using snapshot_download to get flat structure
        # Systran/faster-whisper-large-v3
        repo_id = f"Systran/faster-whisper-{model_size}"
        snapshot_download(repo_id=repo_id, local_dir=output_dir)
        logger.info(f"Faster Whisper {model_size} downloaded to {output_dir}")
        logger.info(f"Directory Contents: {os.listdir(output_dir)}")
    except Exception as e:
        logger.error(f"Failed to download Faster Whisper model: {e}")

def download_pyannote_models():
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not found. Skipping Pyannote model download (requires authentication).")
        return

    logger.info("Downloading Pyannote models...")
    try:
        # Segmentation 3.0
        seg_dir = os.path.join(MODELS_DIR, "pyannote", "segmentation")
        if os.path.exists(seg_dir) and os.listdir(seg_dir):
            logger.info("pyannote/segmentation-3.0 already exists. Skipping.")
        else:
            logger.info("Downloading pyannote/segmentation-3.0...")
            snapshot_download(repo_id="pyannote/segmentation-3.0", local_dir=seg_dir, token=HF_TOKEN)

        # Speaker Diarization 3.1
        diar_dir = os.path.join(MODELS_DIR, "pyannote", "diarization")
        if os.path.exists(diar_dir) and os.listdir(diar_dir):
            logger.info("pyannote/speaker-diarization-3.1 already exists. Skipping.")
        else:
            logger.info("Downloading pyannote/speaker-diarization-3.1...")
            snapshot_download(repo_id="pyannote/speaker-diarization-3.1", local_dir=diar_dir, token=HF_TOKEN)
        
        # Wespeaker Voxceleb
        wespeaker_dir = os.path.join(MODELS_DIR, "pyannote", "wespeaker")
        if os.path.exists(wespeaker_dir) and os.listdir(wespeaker_dir):
            logger.info("pyannote/wespeaker-voxceleb-resnet34-LM already exists. Skipping.")
        else:
            logger.info("Downloading pyannote/wespeaker-voxceleb-resnet34-LM...")
            snapshot_download(repo_id="pyannote/wespeaker-voxceleb-resnet34-LM", local_dir=wespeaker_dir, token=HF_TOKEN)

        logger.info("Pyannote models check/download complete.")
    except Exception as e:
        logger.error(f"Failed to download Pyannote models: {e}")

def download_vad_model():
    cache_dir = os.path.join(MODELS_DIR, "torch_hub")
    # Check if VAD model repo exists in cache, roughly
    if os.path.exists(cache_dir) and any("silero-vad" in d for d in os.listdir(cache_dir)):
        logger.info(f"Silero VAD model seems to exist in {cache_dir}. Skipping download.")
        return

    logger.info("Downloading Silero VAD model...")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        torch.hub.set_dir(cache_dir)
        
        # This will download the repo to {cache_dir}/snakers4_silero-vad_master
        torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, trust_repo=True)
        logger.info(f"Silero VAD model downloaded to {cache_dir}")
    except Exception as e:
        logger.error(f"Failed to download Silero VAD model: {e}")

def download_alignment_model(lang_code="he"):
    # Let's specify the model ID directly as per requirement
    model_id = "imvladikon/wav2vec2-xls-r-300m-hebrew"
    output_dir = os.path.join(MODELS_DIR, "alignment", model_id)

    if os.path.exists(output_dir) and os.listdir(output_dir):
        logger.info(f"Alignment model {model_id} already exists at {output_dir}. Skipping.")
        return

    logger.info(f"Downloading Alignment model for {lang_code}...")
    try:
        snapshot_download(repo_id=model_id, local_dir=output_dir)
        logger.info(f"Alignment model {model_id} downloaded to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to download alignment model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models for WhisperX application.")
    parser.add_argument("--clean", action="store_true", help="Clean existing models directory before downloading.")
    args = parser.parse_args()

    if args.clean and os.path.exists(MODELS_DIR):
        logger.info(f"Cleaning existing models directory: {MODELS_DIR}")
        shutil.rmtree(MODELS_DIR)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    download_faster_whisper()
    download_pyannote_models()
    download_vad_model()
    download_alignment_model()
    
    logger.info("All models downloaded successfully.")
    sys.exit(0)
