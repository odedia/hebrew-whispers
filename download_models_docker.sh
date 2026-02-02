#!/bin/bash

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please export it: export HF_TOKEN=hf_..."
    exit 1
fi

echo "Detected Python 3.13+ or incompatible environment."
echo "Running model download inside a generic Python 3.10 Docker container..."

# Pull the image first to show progress
echo "Pulling python:3.10 image..."
docker pull python:3.10

# Run with Google DNS to avoid network issues
# Removed --network host as it can be flaky on Mac
docker run --rm \
    --dns 8.8.8.8 \
    -v "$(pwd)":/app \
    -w /app \
    -e HF_TOKEN="$HF_TOKEN" \
    python:3.10 \
    /bin/bash -c "
    echo '--- Starting setup inside Docker ---'
    echo '1. Installing dependencies (this may take a few minutes)...'
    pip install huggingface_hub faster-whisper torch whisperx
    
    echo '2. Starting model download script (idempotent)...'
    python download_models.py "$@"
    
    echo '--- Done ---'
    "

echo "Download process finished."
