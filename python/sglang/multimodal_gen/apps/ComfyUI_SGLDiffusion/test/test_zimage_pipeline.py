"""Test for ZImagePipeline with pass-through scheduler."""

import os
import sys

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.test.passthrough import (
    check_passthrough_output,
    prepare_passthrough_request,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request


def test_comfyui_zimage_pipeline_direct() -> None:
    """Test ZImagePipeline with custom inputs."""
    model_path = os.environ.get(
        "SGLANG_TEST_ZIMAGE_MODEL_PATH",
        "Tongyi-MAI/Z-Image-Turbo",  # Supports both safetensors file and diffusers format
    )

    generator = DiffGenerator.from_pretrained(
        model_path=model_path,
        pipeline_class_name="ZImagePipeline",
        num_gpus=1,
        sp_degree=1,
        comfyui_mode=True,
    )

    batch_size = 1
    num_channels = 16
    num_frames = 1
    height = 720
    width = 1280
    latent_height = height // 8
    latent_width = width // 8

    latents = torch.ones(
        batch_size,
        num_channels,
        num_frames,
        latent_height,
        latent_width,
        device="cuda",
        dtype=torch.bfloat16,
    )

    timesteps = torch.tensor([1000], dtype=torch.long, device="cuda")

    context_seq_len = 19
    context_dim = 2560
    context = torch.ones(
        context_seq_len,
        context_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    sampling_params = SamplingParams.from_user_sampling_params_args(
        generator.server_args.model_path,
        server_args=generator.server_args,
        prompt="a beautiful girl",
        guidance_scale=1.0,
        height=height,
        width=width,
        num_frames=1,
        num_inference_steps=1,
        seed=42,
        save_output=False,
        return_frames=False,
    )

    req = prepare_request(
        server_args=generator.server_args,
        sampling_params=sampling_params,
    )

    req.latents = latents
    req.timesteps = timesteps
    req.prompt_embeds = [context]
    req.prompt_seq_lens = [[context_seq_len]]
    req.negative_prompt_embeds = None
    req.raw_latent_shape = torch.tensor(latents.shape, dtype=torch.long)

    prepare_passthrough_request(req)

    check_passthrough_output(generator, req)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
