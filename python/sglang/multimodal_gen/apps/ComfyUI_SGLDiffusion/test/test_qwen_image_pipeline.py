"""Test for QwenImagePipeline with pass-through scheduler."""

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


def test_comfyui_qwen_image_pipeline_direct() -> None:
    """Test QwenImagePipeline with custom inputs."""
    model_path = os.environ.get(
        "SGLANG_TEST_QWEN_IMAGE_MODEL_PATH",
        "Qwen/Qwen-Image",  # Supports both safetensors file and diffusers format
    )

    generator = DiffGenerator.from_pretrained(
        model_path=model_path,
        pipeline_class_name="QwenImagePipeline",
        num_gpus=2,
        comfyui_mode=True,
        dit_layerwise_offload=False,
    )

    batch_size = 1
    hidden_states_seq_len = 6889
    hidden_states_dim = 64
    encoder_seq_len = 45
    encoder_dim = 3584
    height = 1328
    width = 1328
    dtype = torch.bfloat16

    hidden_states = torch.ones(
        batch_size,
        hidden_states_seq_len,
        hidden_states_dim,
        device="cuda",
        dtype=dtype,
    )

    encoder_hidden_states = torch.ones(
        batch_size,
        encoder_seq_len,
        encoder_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    timesteps = torch.tensor([1000], dtype=torch.long, device="cuda")

    sampling_params = SamplingParams.from_user_sampling_params_args(
        generator.server_args.model_path,
        server_args=generator.server_args,
        prompt=" ",
        guidance_scale=3.0,
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

    req.latents = hidden_states
    req.timesteps = timesteps
    req.prompt_embeds = [encoder_hidden_states]
    req.negative_prompt_embeds = [encoder_hidden_states]
    req.raw_latent_shape = torch.tensor(hidden_states.shape, dtype=torch.long)

    prepare_passthrough_request(req)

    check_passthrough_output(generator, req)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
