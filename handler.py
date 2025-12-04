"""
RunPod Serverless Handler for IC-Light

This handler processes image relighting requests via the RunPod serverless platform.
It accepts base64-encoded images and returns base64-encoded results.
"""

import os
import base64
import io
import traceback
from typing import Any

import numpy as np
from PIL import Image

# Import RunPod
import runpod

# Global flag to track model initialization
models_loaded = False


def load_models():
    """Load IC-Light models. Called once at startup."""
    global models_loaded
    if models_loaded:
        return

    # Import gradio_demo module to trigger model loading as a side effect.
    # The module loads models at import time (tokenizer, text_encoder, vae, unet, rmbg).
    # This deferred import allows the handler to start quickly and load models on first request.
    import gradio_demo  # noqa: F401 - import triggers model initialization

    models_loaded = True
    print("Models loaded successfully")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to a numpy array.

    Args:
        base64_string: Base64-encoded image data

    Returns:
        numpy array of the image (H, W, C) in RGB format
    """
    # Handle data URI format
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    return np.array(image)


def encode_image_to_base64(image_array: np.ndarray, format: str = "PNG") -> str:
    """
    Encode a numpy array image to base64 string.

    Args:
        image_array: numpy array of the image (H, W, C)
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64-encoded string of the image
    """
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod serverless handler function.

    Accepts job input with the following parameters:
    - input_image (required): Base64-encoded input image
    - prompt (required): Text prompt for relighting
    - image_width (optional): Output image width (default: 512)
    - image_height (optional): Output image height (default: 640)
    - num_samples (optional): Number of output images (default: 1)
    - seed (optional): Random seed (default: 12345)
    - steps (optional): Number of inference steps (default: 25)
    - a_prompt (optional): Additional prompt (default: 'best quality')
    - n_prompt (optional): Negative prompt (default: 'lowres, bad anatomy, bad hands, cropped, worst quality')
    - cfg (optional): CFG scale (default: 2.0)
    - highres_scale (optional): High-res scale factor (default: 1.5)
    - highres_denoise (optional): High-res denoise strength (default: 0.5)
    - lowres_denoise (optional): Low-res denoise strength (default: 0.9)
    - bg_source (optional): Background source - 'None', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light' (default: 'None')

    Returns:
        Dictionary with:
        - preprocessed_foreground: Base64-encoded preprocessed foreground image
        - results: List of base64-encoded result images
    """
    try:
        # Load models on first request
        load_models()

        # Import after models are loaded
        from gradio_demo import process_relight, BGSource

        # Get job input
        job_input = job.get("input", {})

        # Validate required fields
        if "input_image" not in job_input:
            return {"error": "Missing required field: input_image"}
        if "prompt" not in job_input:
            return {"error": "Missing required field: prompt"}

        # Decode input image
        try:
            input_fg = decode_base64_image(job_input["input_image"])
        except Exception as e:
            return {"error": f"Failed to decode input_image: {str(e)}"}

        # Get parameters with defaults matching the original Gradio demo
        prompt = job_input.get("prompt", "")
        image_width = int(job_input.get("image_width", 512))
        image_height = int(job_input.get("image_height", 640))
        num_samples = int(job_input.get("num_samples", 1))
        seed = int(job_input.get("seed", 12345))
        steps = int(job_input.get("steps", 25))
        a_prompt = job_input.get("a_prompt", "best quality")
        n_prompt = job_input.get("n_prompt", "lowres, bad anatomy, bad hands, cropped, worst quality")
        cfg = float(job_input.get("cfg", 2.0))
        highres_scale = float(job_input.get("highres_scale", 1.5))
        highres_denoise = float(job_input.get("highres_denoise", 0.5))
        lowres_denoise = float(job_input.get("lowres_denoise", 0.9))
        bg_source = job_input.get("bg_source", "None")

        # Validate bg_source
        valid_bg_sources = [e.value for e in BGSource]
        if bg_source not in valid_bg_sources:
            return {
                "error": f"Invalid bg_source: {bg_source}. Must be one of: {valid_bg_sources}"
            }

        # Validate dimensions (must be multiples of 64)
        if image_width % 64 != 0:
            image_width = (image_width // 64) * 64
        if image_height % 64 != 0:
            image_height = (image_height // 64) * 64

        # Clamp values to valid ranges
        image_width = max(256, min(1024, image_width))
        image_height = max(256, min(1024, image_height))
        num_samples = max(1, min(12, num_samples))
        steps = max(1, min(100, steps))
        cfg = max(1.0, min(32.0, cfg))
        highres_scale = max(1.0, min(3.0, highres_scale))
        highres_denoise = max(0.1, min(1.0, highres_denoise))
        lowres_denoise = max(0.1, min(1.0, lowres_denoise))

        # Process the image
        preprocessed_fg, results = process_relight(
            input_fg=input_fg,
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            lowres_denoise=lowres_denoise,
            bg_source=bg_source
        )

        # Encode results to base64
        preprocessed_fg_b64 = encode_image_to_base64(preprocessed_fg)
        results_b64 = [encode_image_to_base64(img) for img in results]

        return {
            "preprocessed_foreground": preprocessed_fg_b64,
            "results": results_b64
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# Start the serverless handler
if __name__ == "__main__":
    print("Starting IC-Light RunPod Serverless Handler...")
    runpod.serverless.start({"handler": handler})
