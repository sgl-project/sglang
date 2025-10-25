# SPDX-License-Identifier: Apache-2.0
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from safetensors.torch import load_file

from sgl_diffusion.runtime.models.vaes.wanvae import (
    AutoencoderKLWan as MyWanVAE,
)
from sgl_diffusion.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def test_wan_vae():
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0")
    # Initialize the two model implementations
    path = "/workspace/data/Wan2.1-T2V-1.3B-Diffusers/vae"
    config_path = os.path.join(path, "config.json")
    config = json.load(open(config_path))
    config.pop("_class_name")
    config.pop("_diffusers_version")
    model1 = MyWanVAE(**config).to(torch.bfloat16)

    model2 = AutoencoderKLWan(**config).to(torch.bfloat16)

    loaded = load_file(os.path.join(path, "diffusion_pytorch_model.safetensors"))
    model1.load_state_dict(loaded)
    model2.load_state_dict(loaded)

    # Set both models to eval mode
    model1.eval()
    model2.eval()

    # Move to GPU
    model1 = model1.to(device)
    model2 = model2.to(device)

    # model1.enable_tiling(
    #     tile_sample_min_height=32,
    #     tile_sample_min_width=32,
    #     tile_sample_min_num_frames=8,
    #     tile_sample_stride_height=16,
    #     tile_sample_stride_width=16,
    #     tile_sample_stride_num_frames=4
    # )

    # Create identical inputs for both models
    batch_size = 1

    # Video input [B, C, T, H, W]
    input_tensor = torch.randn(
        batch_size, 3, 81, 32, 32, device=device, dtype=torch.bfloat16
    )
    latent_tensor = torch.randn(
        batch_size, 16, 21, 32, 32, device=device, dtype=torch.bfloat16
    )

    # Disable gradients for inference
    with torch.no_grad():
        # Test encoding
        logger.info("Testing encoding...")
        latent2 = model2.encode(input_tensor).latent_dist.mean
        print("--------------------------------")
        latent1 = model1.encode(input_tensor).mean
        # Check if latents have the same shape
        assert (
            latent1.shape == latent2.shape
        ), f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
        assert (
            latent1.shape == latent2.shape
        ), f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}"
        # Check if latents are similar
        max_diff_encode = torch.max(torch.abs(latent1 - latent2))
        mean_diff_encode = torch.mean(torch.abs(latent1 - latent2))
        logger.info(
            f"Maximum difference between encoded latents: {max_diff_encode.item()}"
        )
        logger.info(
            f"Mean difference between encoded latents: {mean_diff_encode.item()}"
        )
        assert (
            mean_diff_encode < 5e-1
        ), f"Encoded latents differ significantly: mean diff = {mean_diff_encode.item()}"
        # Test decoding
        logger.info("Testing decoding...")
        latent1 = latent2 = latent_tensor
        latents_mean = (
            torch.tensor(model2.config.latents_mean)
            .view(1, model2.config.z_dim, 1, 1, 1)
            .to(latent2.device, latent2.dtype)
        )
        latents_std = 1.0 / torch.tensor(model2.config.latents_std).view(
            1, model2.config.z_dim, 1, 1, 1
        ).to(latent2.device, latent2.dtype)
        latent2 = latent2 / latents_std + latents_mean
        output1 = model1.decode(latent1)
        output2 = model2.decode(latent2).sample
        # Check if outputs have the same shape
        assert (
            output1.shape == output2.shape
        ), f"Output shapes don't match: {output1.shape} vs {output2.shape}"

        # Check if outputs are similar
        max_diff_decode = torch.max(torch.abs(output1 - output2))
        mean_diff_decode = torch.mean(torch.abs(output1 - output2))
        logger.info(
            f"Maximum difference between decoded outputs: {max_diff_decode.item()}"
        )
        logger.info(
            f"Mean difference between decoded outputs: {mean_diff_decode.item()}"
        )
        assert (
            mean_diff_decode < 1e-1
        ), f"Decoded outputs differ significantly: mean diff = {mean_diff_decode.item()}"

    logger.info("Test passed! Both VAE implementations produce similar outputs.")
    logger.info("Test completed successfully")


if __name__ == "__main__":
    test_wan_vae()
