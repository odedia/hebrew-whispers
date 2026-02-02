FROM python:3.10-bullseye

# Install system dependencies
# ffmpeg is required for audio processing
# git is required for installing some python packages from source
# libav* headers are required for building PyAV from source
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Tanzu security requirement) - Created early to use in COPY --chown
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy requirements first to leverage docker cache
COPY --chown=appuser:appuser requirements.txt .

# Pre-install Cython<3.0 to avoid build errors with PyAV
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "Cython<3.0" && \
    pip install --no-cache-dir -r requirements.txt --no-build-isolation

# Create models directory with correct ownership
RUN mkdir -p /app/models && chown appuser:appuser /app/models

# Copy models with ownership - This PREVENTS layer duplication compared to running chown -R later
# We copy models BEFORE app code so that code changes (frequent) don't invalidate the model layer (rare)
COPY --chown=appuser:appuser models /app/models

# Copy the rest of the application with ownership
COPY --chown=appuser:appuser app /app/app
COPY --chown=appuser:appuser download_models.py .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
