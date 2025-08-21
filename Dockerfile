# Build stage
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential gcc g++ make cmake \
    software-properties-common \
    git curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    setuptools wheel Cython pybind11 setuptools-rust

# Install PyTorch
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CTC Forced Aligner
RUN pip install --no-cache-dir git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# Runtime stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/app/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 ffmpeg libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /app/venv /app/venv

# Create directories
RUN mkdir -p /cache/torch /cache/huggingface /app/tmp

# Environment variables
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/huggingface

# Preload model
RUN python -c "from ctc_forced_aligner import load_alignment_model; import torch; device = 'cuda' if torch.cuda.is_available() else 'cpu'; load_alignment_model(device, model_path='facebook/mms-1b-all', dtype=torch.float16 if device == 'cuda' else torch.float32)"

# Copy handler
COPY handler.py /app/handler.py

STOPSIGNAL SIGINT
CMD ["python", "-u", "handler.py"]