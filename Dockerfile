# Use the updated base CUDA image with development tools
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set Work Directory
WORKDIR /app

# ARGs and ENVs
ARG MMS_MODEL=facebook/mms-1b-all
ARG TORCH_HOME=/cache/torch
ARG HF_HOME=/cache/huggingface

# Environment variables
ENV TORCH_HOME=${TORCH_HOME}
ENV HF_HOME=${HF_HOME}
ENV MMS_MODEL=${MMS_MODEL}
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=True
ENV DEBIAN_FRONTEND=noninteractive

# Update, upgrade, install packages and clean up in one layer
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    # Basic Utilities
    bash ca-certificates curl file git ffmpeg \
    # Build tools (temporarily needed)
    build-essential gcc g++ make cmake \
    # Python 3.10 and development headers
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils python3.10-dev && \
    # Audio libraries
    apt-get install -y libsndfile1-dev && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install pip and upgrade it
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py && \
    python3.10 -m pip install --no-cache-dir --upgrade pip

# Install build dependencies
RUN python3.10 -m pip install --no-cache-dir \
    setuptools \
    wheel \
    Cython \
    pybind11 \
    setuptools-rust

# Install PyTorch with CUDA 11.8 support (matching your working environment)
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.7.1+cu118 \
    torchvision==0.22.1+cu118 \
    torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install core scientific packages (matching your working versions)
RUN python3.10 -m pip install --no-cache-dir \
    numpy==2.1.2 \
    scipy==1.16.1

# Install application requirements (matching your working versions)
RUN python3.10 -m pip install --no-cache-dir \
    transformers==4.55.2 \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    requests==2.32.4 \
    runpod==1.7.13 \
    tqdm==4.67.1 \
    tokenizers==0.21.4 \
    safetensors==0.6.2 \
    huggingface-hub==0.34.4 \
    filelock==3.13.1 \
    fsspec==2024.12.0 \
    packaging==25.0 \
    pyyaml==6.0.2 \
    regex==2025.7.34

# Install CTC Forced Aligner
RUN python3.10 -m pip install --no-cache-dir git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# Clean pip cache to save space
RUN python3.10 -m pip cache purge

# Create directories
RUN mkdir -p /cache/torch /cache/huggingface /app/tmp

# Copy handler
COPY handler.py /app/handler.py

# Test the installation works
RUN python3.10 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python3.10 -c "from ctc_forced_aligner import load_alignment_model; print('CTC Forced Aligner imported successfully')" && \
    python3.10 -c "import transformers; print(f'Transformers version: {transformers.__version__}')" && \
    echo "All imports successful!"

# Set Stop signal and CMD
STOPSIGNAL SIGINT
CMD ["python3.10", "-u", "handler.py"]