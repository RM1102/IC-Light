"""
FastAPI REST API for IC-Light Model

This module provides a REST API for the IC-Light image relighting model,
supporting both foreground conditioning (FC) and foreground-background conditioning (FBC) modes.
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration (configurable via environment variables)
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Import handler functions
from handler import (
    load_models,
    decode_base64_image,
    encode_image_to_base64,
    run_rmbg,
    process_fc,
    process_fbc,
)


class RelightRequest(BaseModel):
    """Request model for relighting endpoint."""

    image: str = Field(..., description="Base64 encoded input image")
    prompt: str = Field(..., description="Text prompt for relighting")
    mode: str = Field(default="fc", description="Model mode: 'fc' or 'fbc'")
    background_image: Optional[str] = Field(
        default=None, description="Base64 encoded background image (required for fbc mode)"
    )
    bg_source: str = Field(default="Left Light", description="Lighting direction")
    width: int = Field(default=512, ge=256, le=1024, description="Output image width")
    height: int = Field(default=640, ge=256, le=1024, description="Output image height")
    steps: Optional[int] = Field(default=None, ge=1, le=100, description="Inference steps")
    seed: int = Field(default=12345, description="Random seed")
    cfg_scale: Optional[float] = Field(default=None, ge=1.0, le=32.0, description="CFG scale")
    highres_scale: float = Field(default=1.5, ge=1.0, le=3.0, description="Highres scale")
    highres_denoise: float = Field(default=0.5, ge=0.1, le=1.0, description="Highres denoise")
    lowres_denoise: float = Field(
        default=0.9, ge=0.1, le=1.0, description="Lowres denoise (fc mode only)"
    )
    num_samples: int = Field(default=1, ge=1, le=12, description="Number of samples")
    a_prompt: str = Field(default="best quality", description="Additional positive prompt")
    n_prompt: str = Field(
        default="lowres, bad anatomy, bad hands, cropped, worst quality",
        description="Negative prompt",
    )
    remove_background: bool = Field(default=True, description="Remove background from input")


class RelightResponse(BaseModel):
    """Response model for relighting endpoint."""

    images: list[str] = Field(..., description="List of base64 encoded result images")
    preprocessed_foreground: Optional[str] = Field(
        default=None, description="Base64 encoded preprocessed foreground"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    model_loaded: bool
    gpu_available: bool


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting IC-Light API...")
    yield
    # Shutdown
    logger.info("Shutting down IC-Light API...")


# Create FastAPI app
app = FastAPI(
    title="IC-Light API",
    description="REST API for IC-Light image relighting model",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    import torch

    from handler import _models_loaded

    return HealthResponse(
        status="healthy",
        model_loaded=_models_loaded,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "IC-Light API",
        "version": "1.0.0",
        "description": "REST API for IC-Light image relighting model",
        "endpoints": {
            "/health": "Health check",
            "/relight": "POST - Relight an image",
            "/docs": "Swagger documentation",
        },
    }


@app.post("/relight", response_model=RelightResponse)
async def relight(request: RelightRequest):
    """
    Relight an image using IC-Light model.

    Supports two modes:
    - fc (Foreground Conditioning): Text-conditioned relighting with lighting direction
    - fbc (Foreground-Background Conditioning): Relighting with custom background image
    """
    try:
        # Validate mode
        if request.mode not in ["fc", "fbc"]:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

        # Validate FBC mode requirements
        if request.mode == "fbc" and not request.background_image:
            raise HTTPException(
                status_code=400, detail="background_image is required for fbc mode"
            )

        # Load models
        load_models(request.mode)

        # Decode input image
        input_fg = decode_base64_image(request.image)

        # Ensure dimensions are multiples of 64
        image_width = int(round(request.width / 64.0) * 64)
        image_height = int(round(request.height / 64.0) * 64)

        # Set default values based on mode
        steps = request.steps if request.steps else (25 if request.mode == "fc" else 20)
        cfg = request.cfg_scale if request.cfg_scale else (2.0 if request.mode == "fc" else 7.0)

        # Remove background if requested
        preprocessed_fg = None
        if request.remove_background:
            input_fg, _ = run_rmbg(input_fg)
            preprocessed_fg = encode_image_to_base64(input_fg)

        # Process based on mode
        if request.mode == "fc":
            results = process_fc(
                input_fg=input_fg,
                prompt=request.prompt,
                image_width=image_width,
                image_height=image_height,
                num_samples=request.num_samples,
                seed=request.seed,
                steps=steps,
                a_prompt=request.a_prompt,
                n_prompt=request.n_prompt,
                cfg=cfg,
                highres_scale=request.highres_scale,
                highres_denoise=request.highres_denoise,
                lowres_denoise=request.lowres_denoise,
                bg_source=request.bg_source,
            )
        else:  # fbc mode
            input_bg = decode_base64_image(request.background_image)

            results = process_fbc(
                input_fg=input_fg,
                input_bg=input_bg,
                prompt=request.prompt,
                image_width=image_width,
                image_height=image_height,
                num_samples=request.num_samples,
                seed=request.seed,
                steps=steps,
                a_prompt=request.a_prompt,
                n_prompt=request.n_prompt,
                cfg=cfg,
                highres_scale=request.highres_scale,
                highres_denoise=request.highres_denoise,
                bg_source=request.bg_source,
            )

        # Encode results
        encoded_results = [encode_image_to_base64(img) for img in results]

        return RelightResponse(
            images=encoded_results,
            preprocessed_foreground=preprocessed_fg,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing relight request")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    logger.info(f"Starting IC-Light API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
