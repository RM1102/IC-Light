# IC-Light RunPod Pod Deployment Dockerfile
# Runs the Gradio demo for interactive use

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Clone the IC-Light repository
RUN git clone https://github.com/lllyasviel/IC-Light.git . || true

# Copy local files (in case building from local context)
COPY . .

# Install PyTorch with CUDA 12.1 support
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
RUN pip install -r requirements.txt

# Create models directory
RUN mkdir -p models

# Expose Gradio port
EXPOSE 7860

# Set the entrypoint to run the Gradio demo
CMD ["python", "gradio_demo.py"]
