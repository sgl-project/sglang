# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from safetensors.torch import load_file

from sgl_diffusion.api.configs.models.vaes import HunyuanVAEConfig
from sgl_diffusion.api.configs.pipelines import PipelineConfig
from sgl_diffusion.runtime.loader.component_loader import VAELoader

# from sgl_diffusion.runtime.models.vaes.hunyuanvae import (
#     AutoencoderKLHunyuanVideo as MyHunyuanVAE)
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.logging_utils import init_logger
from sgl_diffusion.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "hunyuanvideo-community/HunyuanVideo"
MODEL_PATH = maybe_download_model(
    BASE_MODEL_PATH, local_dir=os.path.join("data", BASE_MODEL_PATH)
)
VAE_PATH = os.path.join(MODEL_PATH, "vae")
CONFIG_PATH = os.path.join(VAE_PATH, "config.json")

# Latent generated on commit d71a4ebffc2034922fc379568b6a6aa722f3744c with 1 x A40
# torch 2.7.1
A40_REFERENCE_LATENT = -106.22467041015625

# Latent generated on commit 2b54068960c41d42221e8b8719a374b499855029 with 1 x L40S
L40S_REFERENCE_LATENT = -158.32318115234375


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuan_vae():
    device = torch.device("cuda:0")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = ServerArgs(
        model_path=VAE_PATH,
        pipeline_config=PipelineConfig(
            vae_config=HunyuanVAEConfig(), vae_precision=precision_str
        ),
    )
    args.device = device
    args.vae_cpu_offload = False

    loader = VAELoader()
    model = loader.load(VAE_PATH, args)

    model.enable_tiling(
        tile_sample_min_height=32,
        tile_sample_min_width=32,
        tile_sample_min_num_frames=8,
        tile_sample_stride_height=16,
        tile_sample_stride_width=16,
        tile_sample_stride_num_frames=4,
    )

    batch_size = 1

    # Video input [B, C, T, H, W]
    input_tensor = torch.randn(
        batch_size, 3, 21, 64, 64, device=device, dtype=torch.bfloat16
    )

    # Disable gradients for inference
    with torch.no_grad():
        latent = model.encode(input_tensor).mean.double().sum().item()

    # Check if latents are similar
    device_name = torch.cuda.get_device_name()
    if "A40" in device_name:
        REFERENCE_LATENT = A40_REFERENCE_LATENT
    elif "L40S" in device_name:
        REFERENCE_LATENT = L40S_REFERENCE_LATENT
    else:
        raise ValueError(f"Unknown device: {device_name}")

    diff_encoded_latents = abs(REFERENCE_LATENT - latent)
    logger.info(f"Reference latent: {REFERENCE_LATENT}, Current latent: {latent}")
    assert (
        diff_encoded_latents < 1e-4
    ), f"Encoded latents differ significantly: max diff = {diff_encoded_latents}"
