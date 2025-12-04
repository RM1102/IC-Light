#!/bin/bash
# IC-Light Startup Script
# Downloads model weights if not present and starts the handler

set -e

echo "=========================================="
echo "IC-Light Startup Script"
echo "=========================================="

# Set default values
MODEL_DIR="${MODEL_DIR:-./models}"
RUN_MODE="${RUN_MODE:-handler}"

# Create models directory
mkdir -p "$MODEL_DIR"

# Function to download model if not present
download_model() {
    local model_name=$1
    local model_url=$2
    local model_path="$MODEL_DIR/$model_name"

    if [ ! -f "$model_path" ]; then
        echo "Downloading $model_name..."
        wget -q --show-progress --timeout=30 --tries=3 -O "$model_path" "$model_url"
        echo "Downloaded $model_name successfully."
    else
        echo "$model_name already exists, skipping download."
    fi
}

# Download IC-Light models
echo ""
echo "Checking IC-Light models..."
download_model "iclight_sd15_fc.safetensors" "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors"
download_model "iclight_sd15_fbc.safetensors" "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors"

echo ""
echo "=========================================="
echo "Starting IC-Light in $RUN_MODE mode..."
echo "=========================================="

# Start the appropriate service
case "$RUN_MODE" in
    "handler")
        echo "Starting RunPod handler..."
        exec python -u handler.py
        ;;
    "api")
        echo "Starting FastAPI server..."
        exec python -u api.py
        ;;
    "gradio")
        echo "Starting Gradio demo (FC mode)..."
        exec python -u gradio_demo.py
        ;;
    "gradio-bg")
        echo "Starting Gradio demo (FBC mode)..."
        exec python -u gradio_demo_bg.py
        ;;
    *)
        echo "Unknown run mode: $RUN_MODE"
        echo "Valid modes: handler, api, gradio, gradio-bg"
        exit 1
        ;;
esac
