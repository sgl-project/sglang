# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from sgl_diffusion.api.configs.models.vaes import WanVAEConfig
from sgl_diffusion.api.configs.pipelines import PipelineConfig
from sgl_diffusion.runtime.loader.component_loader import VAELoader
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.hf_diffusers_utils import maybe_download_model
from sgl_diffusion.runtime.utils.logging_utils import init_logger

try:
    from wan.modules.vae2_2 import Wan2_2_VAE
except ImportError:
    Wan2_2_VAE = None

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH)
VAE_PATH = os.path.join(MODEL_PATH, "vae")


@pytest.mark.skip(reason="disable test")
@pytest.mark.usefixtures("distributed_setup")
def test_official_wan_vae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = ServerArgs(
        model_path=VAE_PATH,
        pipeline_config=PipelineConfig(
            vae_config=WanVAEConfig(), vae_precision=precision_str
        ),
    )
    args.device = device
    args.vae_cpu_offload = False

    loader = VAELoader()
    model2 = loader.load(VAE_PATH, args)
    assert model2.use_feature_cache  # Default to use the original WanVAE algorithm

    model1 = Wan2_2_VAE(
        vae_pth="/mnt/weka/home/hao.zhang/wei/Wan2.2_VAE.pth",
        device=device,
        dtype=precision,
    )

    # Create identical inputs for both models
    batch_size = 1

    # Video input [B, C, T, H, W]
    input_tensor = torch.randn(
        batch_size, 3, 81, 32, 32, device=device, dtype=precision
    )
    # latent_tensor = torch.randn(batch_size,
    #                             16,
    #                             21,
    #                             32,
    #                             32,
    #                             device=device,
    #                             dtype=precision)

    # Disable gradients for inference
    with torch.no_grad():
        # Test encoding
        logger.info("Testing encoding...")
        latent1 = model1.encode([input_tensor.squeeze(0)])[0].unsqueeze(0)
        print("--------------------------------")
        latent2 = model2.encode(input_tensor)
        # Check if latents have the same shape
        assert (
            latent1.shape == latent2.mean.shape
        ), f"Latent shapes don't match: {latent1.mean.shape} vs {latent2.mean.shape}"
        # Check if latents are similar
        max_diff_encode = torch.max(torch.abs(latent1 - latent2.mean))
        mean_diff_encode = torch.mean(torch.abs(latent1 - latent2.mean))
        logger.info(
            "Maximum difference between encoded latents: %s", max_diff_encode.item()
        )
        logger.info(
            "Mean difference between encoded latents: %s", mean_diff_encode.item()
        )
        assert (
            max_diff_encode < 1e-5
        ), f"Encoded latents differ significantly: max diff = {mean_diff_encode.item()}"
        # Test decoding
        logger.info("Testing decoding...")
        # latent1_tensor = latent1.mode()
        # mean1 = (torch.tensor(model1.config.latents_mean).view(
        #     1, model1.config.z_dim, 1, 1, 1).to(input_tensor.device,
        #                                         input_tensor.dtype))
        # std1 = (1.0 / torch.tensor(model1.config.latents_std).view(
        #     1, model1.config.z_dim, 1, 1, 1)).to(input_tensor.device,
        #                                         input_tensor.dtype)
        # latent1_tensor = latent1_tensor / std1 + mean1
        # output1 = model1.decode(latent1_tensor).sample

        output1 = model1.decode([latent1.squeeze(0)])
        output1 = output1[0].unsqueeze(0)

        mean2 = model2.config.arch_config.shift_factor.to(
            input_tensor.device, input_tensor.dtype
        )
        std2 = model2.config.arch_config.scaling_factor.to(
            input_tensor.device, input_tensor.dtype
        )
        latent2_tensor = latent2.mode()
        latent2_tensor = latent2_tensor / std2 + mean2
        output2 = model2.decode(latent2_tensor)
        # Check if outputs have the same shape
        assert (
            output1.shape == output2.shape
        ), f"Output shapes don't match: {output1.shape} vs {output2.shape}"

        # Check if outputs are similar
        max_diff_decode = torch.max(torch.abs(output1.float() - output2.float()))
        mean_diff_decode = torch.mean(torch.abs(output1.float() - output2.float()))
        logger.info(
            "Maximum difference between decoded outputs: %s", max_diff_decode.item()
        )
        logger.info(
            "Mean difference between decoded outputs: %s", mean_diff_decode.item()
        )
        assert (
            max_diff_decode < 1e-5
        ), f"Decoded outputs differ significantly: max diff = {max_diff_decode.item()}"
