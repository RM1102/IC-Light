# IC-Light RunPod Deployment Guide

This guide provides comprehensive instructions for deploying IC-Light on RunPod infrastructure, supporting both Pod deployment (with interactive Gradio UI) and Serverless deployment (for API-based inference).

## Table of Contents

- [Prerequisites](#prerequisites)
- [GPU Requirements](#gpu-requirements)
- [Pod Deployment](#pod-deployment)
- [Serverless Deployment](#serverless-deployment)
- [API Reference](#api-reference)
- [Example API Calls](#example-api-calls)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying IC-Light on RunPod, ensure you have:

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Docker**: Installed locally for building images
3. **Docker Hub Account** (or other container registry): For storing Docker images
4. **Git**: For cloning the repository

## GPU Requirements

IC-Light requires significant GPU memory for image generation. We recommend:

| GPU | VRAM | Recommended |
|-----|------|-------------|
| NVIDIA RTX 3090 | 24GB | ✅ Yes |
| NVIDIA RTX 4090 | 24GB | ✅ Yes |
| NVIDIA A100 | 40GB/80GB | ✅ Yes |
| NVIDIA A40 | 48GB | ✅ Yes |
| NVIDIA L40 | 48GB | ✅ Yes |
| NVIDIA RTX 4080 | 16GB | ⚠️ May work with smaller images |
| NVIDIA RTX 3080 | 10GB | ❌ Not recommended |

**Minimum VRAM**: 16GB (with reduced image sizes)
**Recommended VRAM**: 24GB or more

---

## Pod Deployment

Pod deployment runs the interactive Gradio UI, allowing you to use IC-Light through a web browser.

### Step 1: Build the Docker Image

```bash
# Clone the repository
git clone https://github.com/lllyasviel/IC-Light.git
cd IC-Light

# Build the Docker image
docker build -t your-dockerhub-username/iclight-pod:latest -f Dockerfile .

# Push to Docker Hub
docker push your-dockerhub-username/iclight-pod:latest
```

### Step 2: Create a RunPod Pod

1. Log in to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **"+ Deploy"**
3. Select a GPU with at least 24GB VRAM (RTX 3090, RTX 4090, A100, etc.)
4. Under **"Container Image"**, enter: `your-dockerhub-username/iclight-pod:latest`
5. Set **"Container Disk"** to at least 50GB
6. Set **"Volume Disk"** to at least 20GB (for model caching)
7. Configure **"Expose HTTP Ports"**: `7860`
8. Click **"Deploy"**

### Step 3: Access the Gradio UI

1. Wait for the pod to start (may take 5-10 minutes for initial model download)
2. Click **"Connect"** on your pod
3. Select **"Connect via HTTP Service [Port 7860]"**
4. The Gradio UI will open in your browser

---

## Serverless Deployment

Serverless deployment provides an API endpoint for programmatic access, with automatic scaling and pay-per-use pricing.

### Step 1: Build the Serverless Docker Image

```bash
# Clone the repository (if not already done)
git clone https://github.com/lllyasviel/IC-Light.git
cd IC-Light

# Build the serverless Docker image
docker build -t your-dockerhub-username/iclight-serverless:latest -f Dockerfile.serverless .

# Push to Docker Hub
docker push your-dockerhub-username/iclight-serverless:latest
```

### Step 2: Create a RunPod Serverless Endpoint

1. Log in to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure the endpoint:
   - **Endpoint Name**: `iclight-api`
   - **Select Template**: Choose **"Custom"**
   - **Container Image**: `your-dockerhub-username/iclight-serverless:latest`
   - **Container Disk**: 50GB
   - **GPU Type**: Select GPUs with 24GB+ VRAM
   - **Max Workers**: Set based on your expected load (start with 1)
   - **Idle Timeout**: 60 seconds (adjust based on usage patterns)
   - **Active Workers**: 0 (scale to zero when idle)
4. Click **"Create"**

### Step 3: Get Your API Credentials

1. Navigate to your endpoint in the RunPod console
2. Copy your **Endpoint ID** (e.g., `abc123xyz`)
3. Get your **API Key** from [Settings > API Keys](https://www.runpod.io/console/user/settings)

### Step 4: Test the Endpoint

Use the provided Python client or curl:

```bash
# Set environment variables
export RUNPOD_API_KEY=your_api_key
export RUNPOD_ENDPOINT_ID=your_endpoint_id

# Run the example client
python runpod_client.py
```

---

## API Reference

### Endpoint URL

```
https://api.runpod.ai/v2/{endpoint_id}/runsync  # Synchronous
https://api.runpod.ai/v2/{endpoint_id}/run      # Asynchronous
```

### Authentication

Include your API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_image` | string | ✅ Yes | - | Base64-encoded input image |
| `prompt` | string | ✅ Yes | - | Text prompt for relighting |
| `image_width` | integer | No | 512 | Output width (256-1024, multiple of 64) |
| `image_height` | integer | No | 640 | Output height (256-1024, multiple of 64) |
| `num_samples` | integer | No | 1 | Number of output images (1-12) |
| `seed` | integer | No | 12345 | Random seed for reproducibility |
| `steps` | integer | No | 25 | Number of inference steps (1-100) |
| `a_prompt` | string | No | "best quality" | Additional positive prompt |
| `n_prompt` | string | No | "lowres, bad anatomy, bad hands, cropped, worst quality" | Negative prompt |
| `cfg` | float | No | 2.0 | CFG scale (1.0-32.0) |
| `highres_scale` | float | No | 1.5 | High-res scale factor (1.0-3.0) |
| `highres_denoise` | float | No | 0.5 | High-res denoise strength (0.1-1.0) |
| `lowres_denoise` | float | No | 0.9 | Low-res denoise strength (0.1-1.0) |
| `bg_source` | string | No | "None" | Lighting preference |

### Background Source Options

| Value | Description |
|-------|-------------|
| `"None"` | No initial lighting preference |
| `"Left Light"` | Light from the left side |
| `"Right Light"` | Light from the right side |
| `"Top Light"` | Light from above |
| `"Bottom Light"` | Light from below |

### Response Format

**Success Response:**

```json
{
  "status": "COMPLETED",
  "output": {
    "preprocessed_foreground": "base64_encoded_image...",
    "results": [
      "base64_encoded_image_1...",
      "base64_encoded_image_2..."
    ]
  }
}
```

**Error Response:**

```json
{
  "status": "FAILED",
  "error": "Error message description"
}
```

---

## Example API Calls

### Using cURL (Synchronous)

```bash
# Encode your image to base64
IMAGE_B64=$(base64 -i your_image.jpg)

# Make the API call
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input_image": "'"${IMAGE_B64}"'",
      "prompt": "beautiful woman, detailed face, sunshine from window",
      "image_width": 512,
      "image_height": 640,
      "seed": 12345,
      "bg_source": "Left Light"
    }
  }'
```

### Using cURL (Asynchronous)

```bash
# Start the job
JOB_RESPONSE=$(curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input_image": "'"${IMAGE_B64}"'",
      "prompt": "beautiful woman, detailed face, neon light",
      "bg_source": "Right Light"
    }
  }')

# Extract job ID
JOB_ID=$(echo $JOB_RESPONSE | jq -r '.id')

# Check status
curl -X GET "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

### Using Python

```python
import os
import base64
import requests

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

# Encode image
with open("your_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Make request
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "input_image": image_b64,
            "prompt": "beautiful woman, sunshine from window",
            "image_width": 512,
            "image_height": 640,
            "seed": 12345,
            "bg_source": "Left Light"
        }
    },
    timeout=300
)

result = response.json()

if result.get("status") == "COMPLETED":
    # Decode and save results
    for i, img_b64 in enumerate(result["output"]["results"]):
        img_data = base64.b64decode(img_b64)
        with open(f"result_{i}.png", "wb") as f:
            f.write(img_data)
```

---

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory" Error

**Cause**: Insufficient GPU VRAM

**Solutions**:
- Use a GPU with more VRAM (24GB+ recommended)
- Reduce `image_width` and `image_height`
- Reduce `num_samples` to 1
- Reduce `highres_scale` to 1.0

#### 2. Slow Cold Start

**Cause**: Models need to be downloaded on first run

**Solutions**:
- Use the serverless image with pre-downloaded models
- Increase the idle timeout to keep workers warm
- Set `Active Workers` to 1 to keep a warm instance

#### 3. "Model not found" Error

**Cause**: Model files not downloaded

**Solutions**:
- Wait for automatic model download on first request
- Manually download models by connecting to the pod

#### 4. Timeout Errors

**Cause**: Processing takes longer than expected

**Solutions**:
- Use asynchronous API calls instead of synchronous
- Increase timeout values in your client code
- Reduce image dimensions or number of samples

#### 5. "Invalid bg_source" Error

**Cause**: Incorrect lighting preference value

**Solution**: Use one of the valid values:
- `"None"`
- `"Left Light"`
- `"Right Light"`
- `"Top Light"`
- `"Bottom Light"`

### Getting Help

1. Check the [RunPod Documentation](https://docs.runpod.io/)
2. Visit the [IC-Light GitHub Issues](https://github.com/lllyasviel/IC-Light/issues)
3. Join the [RunPod Discord](https://discord.gg/runpod)

### Logs and Debugging

**For Pods:**
```bash
# View logs via RunPod console
# Click on your pod > Logs
```

**For Serverless:**
```bash
# View logs in the RunPod console
# Navigate to your endpoint > Logs tab
```

---

## Cost Optimization Tips

1. **Use Serverless**: Pay only for actual compute time
2. **Set Idle Timeout**: Configure appropriate timeout to balance cost and cold starts
3. **Batch Requests**: Process multiple images in sequence to keep workers warm
4. **Choose Right GPU**: RTX 3090 offers good balance of performance and cost
5. **Monitor Usage**: Use RunPod dashboard to track spending

---

## Security Considerations

1. **Keep API Keys Secure**: Never commit API keys to version control
2. **Use Environment Variables**: Store sensitive data in environment variables
3. **Limit Access**: Use RunPod's team features to control who can access endpoints
4. **Monitor Usage**: Watch for unusual activity that could indicate key compromise
