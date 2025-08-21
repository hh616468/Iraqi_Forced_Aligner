# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set shell to fail on errors
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=True \
    PATH="/app/venv/bin:$PATH" \
    TORCH_HOME=/cache/torch \
    HF_HOME=/cache/huggingface

# System dependencies: Install all in one layer for efficiency
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # >>> ADD THIS PACKAGE <<<
        build-essential \
        # Basic utilities
        git ffmpeg curl ca-certificates \
        # Python 3.10
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    # Audio library dependency
    apt-get install -y libsndfile1-dev && \
    # Clean up to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up the application directory and virtual environment
WORKDIR /app
RUN python3.10 -m venv /app/venv

# Copy requirements file before other files to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# This is more manageable and better for caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==0.2.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install the git package separately
    pip install --no-cache-dir git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git && \
    # Clean pip cache
    pip cache purge

# Create necessary directories
RUN mkdir -p /cache/torch /cache/huggingface /app/tmp

# Copy your application code
COPY handler.py .

# Test the installation to ensure everything is working
RUN python3.10 -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Set the final command to run your application
CMD ["python3.10", "-u", "handler.py"]