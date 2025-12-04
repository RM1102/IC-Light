"""
RunPod Serverless Handler for IC-Light Model

This module provides a serverless handler for the IC-Light image relighting model,
supporting both foreground conditioning (FC) and foreground-background conditioning (FBC) modes.
"""

import os
import math
import base64
import logging
from io import BytesIO
from enum import Enum
from typing import Optional

import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from torch.hub import download_url_to_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs
MODEL_URLS = {
    "fc": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
    "fbc": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors",
}

# Model paths
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
FC_MODEL_PATH = os.path.join(MODEL_DIR, "iclight_sd15_fc.safetensors")
FBC_MODEL_PATH = os.path.join(MODEL_DIR, "iclight_sd15_fbc.safetensors")

# Global model components (lazy loaded)
_models_loaded = False
_current_mode = None
tokenizer = None
text_encoder = None
vae = None
unet = None
rmbg = None
t2i_pipe = None
i2i_pipe = None
device = None


class BGSource(Enum):
    """Background source options for relighting."""

    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    GREY = "Ambient"


def ensure_model_downloaded(mode: str = "fc") -> str:
    """Download model weights if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if mode == "fc":
        model_path = FC_MODEL_PATH
        model_url = MODEL_URLS["fc"]
    else:
        model_path = FBC_MODEL_PATH
        model_url = MODEL_URLS["fbc"]

    if not os.path.exists(model_path):
        logger.info(f"Downloading {mode} model to {model_path}...")
        download_url_to_file(url=model_url, dst=model_path)
        logger.info(f"Model downloaded successfully.")

    return model_path


def load_models(mode: str = "fc"):
    """Load IC-Light models and components."""
    global _models_loaded, _current_mode
    global tokenizer, text_encoder, vae, unet, rmbg, t2i_pipe, i2i_pipe, device

    # Skip if already loaded with same mode
    if _models_loaded and _current_mode == mode:
        return

    logger.info(f"Loading IC-Light models in {mode} mode...")

    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        AutoencoderKL,
        UNet2DConditionModel,
        DPMSolverMultistepScheduler,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTokenizer
    from briarmbg import BriaRMBG

    # Base SD model
    sd15_name = "stablediffusionapi/realistic-vision-v51"

    # Load components
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

    # Modify UNet for IC-Light
    in_channels = 8 if mode == "fc" else 12
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            in_channels,
            unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    # Hook UNet forward
    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        c_concat = torch.cat(
            [c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0
        )
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs["cross_attention_kwargs"] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = hooked_unet_forward

    # Load IC-Light weights
    model_path = ensure_model_downloaded(mode)
    sd_offset = sf.load_file(model_path)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    unet = unet.to(device=device, dtype=torch.float16)
    rmbg = rmbg.to(device=device, dtype=torch.float32)

    # Set attention processors
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    # Create scheduler
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1,
    )

    # Create pipelines
    t2i_pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None,
    )

    i2i_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None,
    )

    _models_loaded = True
    _current_mode = mode
    logger.info(f"Models loaded successfully in {mode} mode.")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 string to numpy array."""
    # Handle data URL format
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return np.array(image)


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy array to base64 string."""
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    """Encode a text prompt to embeddings."""
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [
        [id_start] + tokens[i : i + chunk_length] + [id_end]
        for i in range(0, len(tokens), chunk_length)
    ]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt: str, negative_prompt: str):
    """Encode positive and negative prompts."""
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    """Convert PyTorch tensors to numpy arrays."""
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    """Convert numpy arrays to PyTorch tensors."""
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize and center crop an image."""
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize an image without cropping."""
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img: np.ndarray, sigma: float = 0.0):
    """Remove background from image."""
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process_fc(
    input_fg: np.ndarray,
    prompt: str,
    image_width: int,
    image_height: int,
    num_samples: int,
    seed: int,
    steps: int,
    a_prompt: str,
    n_prompt: str,
    cfg: float,
    highres_scale: float,
    highres_denoise: float,
    lowres_denoise: float,
    bg_source: str,
):
    """Process foreground-conditioned relighting."""
    bg_source_enum = BGSource(bg_source)
    input_bg = None

    if bg_source_enum == BGSource.NONE:
        pass
    elif bg_source_enum == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source_enum == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source_enum == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source_enum == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise ValueError("Invalid background source for FC mode")

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt
    )

    if input_bg is None:
        latents = (
            t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = (
            i2i_pipe(
                image=bg_latent,
                strength=lowres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / lowres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [
        resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64),
        )
        for p in pixels
    ]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = (
        i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype)
        / vae.config.scaling_factor
    )

    pixels = vae.decode(latents).sample
    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_fbc(
    input_fg: np.ndarray,
    input_bg: np.ndarray,
    prompt: str,
    image_width: int,
    image_height: int,
    num_samples: int,
    seed: int,
    steps: int,
    a_prompt: str,
    n_prompt: str,
    cfg: float,
    highres_scale: float,
    highres_denoise: float,
    bg_source: str,
):
    """Process foreground-background-conditioned relighting."""
    bg_source_enum = BGSource(bg_source)

    if bg_source_enum == BGSource.UPLOAD:
        pass
    elif bg_source_enum == BGSource.UPLOAD_FLIP:
        input_bg = np.fliplr(input_bg)
    elif bg_source_enum == BGSource.GREY:
        input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
    elif bg_source_enum == BGSource.LEFT:
        gradient = np.linspace(224, 32, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source_enum == BGSource.RIGHT:
        gradient = np.linspace(32, 224, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source_enum == BGSource.TOP:
        gradient = np.linspace(224, 32, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source_enum == BGSource.BOTTOM:
        gradient = np.linspace(32, 224, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise ValueError("Invalid background source for FBC mode")

    rng = torch.Generator(device=device).manual_seed(seed)

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt
    )

    latents = (
        t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype)
        / vae.config.scaling_factor
    )

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [
        resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64),
        )
        for p in pixels
    ]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    latents = (
        i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype)
        / vae.config.scaling_factor
    )

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)
    return [(x * 255.0).clip(0, 255).astype(np.uint8) for x in pixels]


def handler(job: dict) -> dict:
    """
    RunPod serverless handler function.

    Input format:
    {
        "input": {
            "image": "<base64 encoded image>",
            "prompt": "description of desired lighting",
            "mode": "fc" or "fbc",  # optional, default: "fc"
            "background_image": "<base64 encoded background>",  # required for fbc mode
            "bg_source": "Left Light",  # lighting direction
            "width": 512,  # optional
            "height": 640,  # optional
            "steps": 25,  # optional
            "seed": 12345,  # optional
            "cfg_scale": 2.0,  # optional
            "highres_scale": 1.5,  # optional
            "highres_denoise": 0.5,  # optional
            "lowres_denoise": 0.9,  # optional (fc mode only)
            "num_samples": 1,  # optional
            "a_prompt": "best quality",  # optional
            "n_prompt": "lowres, bad anatomy",  # optional
            "remove_background": true  # optional, default: true
        }
    }

    Output format:
    {
        "images": ["<base64 encoded result image>", ...],
        "preprocessed_foreground": "<base64 encoded foreground>"  # if remove_background is true
    }
    """
    try:
        job_input = job.get("input", {})

        # Get required parameters
        image_b64 = job_input.get("image")
        if not image_b64:
            return {"error": "Missing required parameter: image"}

        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "Missing required parameter: prompt"}

        # Get mode (fc or fbc)
        mode = job_input.get("mode", "fc").lower()
        if mode not in ["fc", "fbc"]:
            return {"error": f"Invalid mode: {mode}. Must be 'fc' or 'fbc'."}

        # Load models
        load_models(mode)

        # Decode input image
        input_fg = decode_base64_image(image_b64)

        # Get optional parameters
        bg_source = job_input.get("bg_source", "Left Light")
        image_width = job_input.get("width", 512)
        image_height = job_input.get("height", 640)
        steps = job_input.get("steps", 25 if mode == "fc" else 20)
        seed = job_input.get("seed", 12345)
        cfg = job_input.get("cfg_scale", 2.0 if mode == "fc" else 7.0)
        highres_scale = job_input.get("highres_scale", 1.5)
        highres_denoise = job_input.get("highres_denoise", 0.5)
        lowres_denoise = job_input.get("lowres_denoise", 0.9)
        num_samples = job_input.get("num_samples", 1)
        a_prompt = job_input.get("a_prompt", "best quality")
        n_prompt = job_input.get(
            "n_prompt", "lowres, bad anatomy, bad hands, cropped, worst quality"
        )
        remove_background = job_input.get("remove_background", True)

        # Ensure dimensions are multiples of 64
        image_width = int(round(image_width / 64.0) * 64)
        image_height = int(round(image_height / 64.0) * 64)

        # Remove background if requested
        preprocessed_fg = None
        if remove_background:
            input_fg, _ = run_rmbg(input_fg)
            preprocessed_fg = encode_image_to_base64(input_fg)

        # Process based on mode
        if mode == "fc":
            results = process_fc(
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
                bg_source=bg_source,
            )
        else:  # fbc mode
            background_b64 = job_input.get("background_image")
            if not background_b64:
                return {"error": "Missing required parameter for FBC mode: background_image"}

            input_bg = decode_base64_image(background_b64)

            results = process_fbc(
                input_fg=input_fg,
                input_bg=input_bg,
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
                bg_source=bg_source,
            )

        # Encode results
        encoded_results = [encode_image_to_base64(img) for img in results]

        response = {"images": encoded_results}
        if preprocessed_fg:
            response["preprocessed_foreground"] = preprocessed_fg

        return response

    except Exception as e:
        logger.exception("Error processing request")
        return {"error": str(e)}


# RunPod serverless entry point
if __name__ == "__main__":
    try:
        import runpod

        runpod.serverless.start({"handler": handler})
    except ImportError:
        logger.warning("RunPod not installed. Running in standalone mode.")
        # For local testing
        print("Handler module loaded successfully. Use RunPod to start the serverless handler.")
