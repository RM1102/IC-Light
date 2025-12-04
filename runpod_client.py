"""
RunPod Client Example for IC-Light

This script demonstrates how to call the IC-Light RunPod serverless endpoint.
It includes examples for both synchronous and asynchronous API calls.

Usage:
    1. Set your RunPod API key: export RUNPOD_API_KEY=your_api_key
    2. Set your endpoint ID: export RUNPOD_ENDPOINT_ID=your_endpoint_id
    3. Run: python runpod_client.py
"""

import os
import base64
import time
import json
from typing import Optional

import requests


def encode_image_file(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded string of the image
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_file(base64_string: str, output_path: str) -> None:
    """
    Decode a base64 string and save it as an image file.

    Args:
        base64_string: Base64-encoded image data
        output_path: Path to save the decoded image
    """
    image_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(image_data)
    print(f"Saved image to: {output_path}")


def run_sync(
    api_key: str,
    endpoint_id: str,
    input_image_path: str,
    prompt: str,
    **kwargs
) -> dict:
    """
    Run IC-Light inference synchronously (blocking).
    Waits for the job to complete and returns the results.

    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        input_image_path: Path to the input image
        prompt: Text prompt for relighting
        **kwargs: Additional parameters (image_width, image_height, seed, etc.)

    Returns:
        Dictionary containing the results or error
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Encode the input image
    input_image_b64 = encode_image_file(input_image_path)

    # Build the payload
    payload = {
        "input": {
            "input_image": input_image_b64,
            "prompt": prompt,
            **kwargs
        }
    }

    print(f"Sending synchronous request to {url}...")
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    return response.json()


def run_async(
    api_key: str,
    endpoint_id: str,
    input_image_path: str,
    prompt: str,
    **kwargs
) -> str:
    """
    Start an IC-Light inference job asynchronously (non-blocking).
    Returns the job ID for later status checking.

    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        input_image_path: Path to the input image
        prompt: Text prompt for relighting
        **kwargs: Additional parameters

    Returns:
        Job ID string
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Encode the input image
    input_image_b64 = encode_image_file(input_image_path)

    # Build the payload
    payload = {
        "input": {
            "input_image": input_image_b64,
            "prompt": prompt,
            **kwargs
        }
    }

    print(f"Sending asynchronous request to {url}...")
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    job_id = result.get("id")
    print(f"Job submitted. Job ID: {job_id}")

    return job_id


def check_status(api_key: str, endpoint_id: str, job_id: str) -> dict:
    """
    Check the status of an async job.

    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        job_id: The job ID to check

    Returns:
        Dictionary containing job status and results if complete
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return response.json()


def wait_for_completion(
    api_key: str,
    endpoint_id: str,
    job_id: str,
    poll_interval: int = 5,
    max_wait: int = 300
) -> dict:
    """
    Poll for job completion and return results.

    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        job_id: The job ID to wait for
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait

    Returns:
        Dictionary containing the final job results
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status = check_status(api_key, endpoint_id, job_id)
        job_status = status.get("status")

        print(f"Job status: {job_status}")

        if job_status == "COMPLETED":
            return status
        elif job_status in ["FAILED", "CANCELLED"]:
            return status

        time.sleep(poll_interval)

    return {"status": "TIMEOUT", "error": f"Job did not complete within {max_wait} seconds"}


def cancel_job(api_key: str, endpoint_id: str, job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        job_id: The job ID to cancel

    Returns:
        Dictionary containing cancellation result
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{job_id}"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, timeout=30)
    response.raise_for_status()

    return response.json()


# Example usage
if __name__ == "__main__":
    # Get API credentials from environment variables
    API_KEY = os.environ.get("RUNPOD_API_KEY")
    ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

    if not API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        print("Set it with: export RUNPOD_API_KEY=your_api_key")
        exit(1)

    if not ENDPOINT_ID:
        print("Error: RUNPOD_ENDPOINT_ID environment variable not set")
        print("Set it with: export RUNPOD_ENDPOINT_ID=your_endpoint_id")
        exit(1)

    # Example: Process an image
    # Try to find an example image, or use a custom path from environment
    INPUT_IMAGE = os.environ.get("INPUT_IMAGE_PATH", "imgs/i1.webp")

    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: Input image not found: {INPUT_IMAGE}")
        print("Please provide a valid image path via INPUT_IMAGE_PATH environment variable")
        print("Example: export INPUT_IMAGE_PATH=/path/to/your/image.jpg")
        exit(1)

    print("\n" + "="*60)
    print("IC-Light RunPod Client Example")
    print("="*60)

    # Example 1: Synchronous call
    print("\n--- Example 1: Synchronous API Call ---")
    try:
        result = run_sync(
            api_key=API_KEY,
            endpoint_id=ENDPOINT_ID,
            input_image_path=INPUT_IMAGE,
            prompt="beautiful woman, detailed face, sunshine from window",
            image_width=512,
            image_height=640,
            seed=12345,
            bg_source="Left Light"
        )

        if result.get("status") == "COMPLETED":
            output = result.get("output", {})

            # Save the preprocessed foreground
            if "preprocessed_foreground" in output:
                decode_base64_to_file(
                    output["preprocessed_foreground"],
                    "output_preprocessed.png"
                )

            # Save the results
            for i, img_b64 in enumerate(output.get("results", [])):
                decode_base64_to_file(img_b64, f"output_result_{i}.png")

            print("Synchronous call completed successfully!")
        else:
            print(f"Job failed: {result}")

    except Exception as e:
        print(f"Synchronous call failed: {e}")

    # Example 2: Asynchronous call
    print("\n--- Example 2: Asynchronous API Call ---")
    try:
        # Start the job
        job_id = run_async(
            api_key=API_KEY,
            endpoint_id=ENDPOINT_ID,
            input_image_path=INPUT_IMAGE,
            prompt="beautiful woman, detailed face, neon light, city",
            image_width=512,
            image_height=640,
            seed=42,
            bg_source="Right Light"
        )

        # Wait for completion
        result = wait_for_completion(
            api_key=API_KEY,
            endpoint_id=ENDPOINT_ID,
            job_id=job_id,
            poll_interval=5,
            max_wait=300
        )

        if result.get("status") == "COMPLETED":
            output = result.get("output", {})

            # Save results
            for i, img_b64 in enumerate(output.get("results", [])):
                decode_base64_to_file(img_b64, f"output_async_result_{i}.png")

            print("Asynchronous call completed successfully!")
        else:
            print(f"Job failed: {result}")

    except Exception as e:
        print(f"Asynchronous call failed: {e}")

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
