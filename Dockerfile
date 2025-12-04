# Production Dockerfile for IC-Light
# Supports RunPod serverless and any Docker-compatible platform

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support (pinned version for reproducible builds)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Download model weights during build (optional - can be done at runtime)
# Uncomment the following lines to pre-download models
# RUN python -c "from torch.hub import download_url_to_file; \
#     download_url_to_file('https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', '/app/models/iclight_sd15_fc.safetensors'); \
#     download_url_to_file('https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', '/app/models/iclight_sd15_fbc.safetensors')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch

# Create cache directories
RUN mkdir -p /app/cache/transformers /app/cache/huggingface /app/cache/torch

# Expose port for API
EXPOSE 8000

# Make start script executable (fail if file doesn't exist)
RUN if [ -f /app/start.sh ]; then chmod +x /app/start.sh; fi

# Default command - runs the RunPod handler
CMD ["python", "-u", "handler.py"]
