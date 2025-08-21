# Use the updated base CUDA image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

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

# Set LD_LIBRARY_PATH for library location (if still necessary)
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=True
ENV DEBIAN_FRONTEND=noninteractive

# Update, upgrade, install packages and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    # Basic Utilities
    bash ca-certificates curl file git ffmpeg \
    # Python 3.10 and venv
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Set locale
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Create and activate virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip==21.*

# Install setuptools-rust and other smaller packages
RUN pip install --no-cache-dir \
    setuptools-rust==1.8.0 \
    huggingface_hub==0.18.0 \
    runpod==1.3.0

# Install PyTorch packages (compatible with CUDA 11.8)
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Copy and install application-specific requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CTC Forced Aligner
RUN pip install --no-cache-dir git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# Create cache directories
RUN mkdir -p /cache/torch /cache/huggingface

# Preload the MMS model for forced alignment
RUN python -c "from ctc_forced_aligner import load_alignment_model; import torch; device = 'cuda' if torch.cuda.is_available() else 'cpu'; load_alignment_model(device, model_path='${MMS_MODEL}', dtype=torch.float16 if device == 'cuda' else torch.float32)"

# Create temp directory for processing
RUN mkdir -p /app/tmp

# Copy the handler
COPY handler.py /app/handler.py

# Set Stop signal and CMD
STOPSIGNAL SIGINT
CMD ["python", "-u", "handler.py"]


