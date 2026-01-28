"""Test for ComfyUIZImagePipeline with pass-through scheduler."""

import os

import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request


def test_comfyui_zimage_pipeline_direct() -> None:
    """Test ComfyUIZImagePipeline with custom inputs."""
    model_path = os.environ.get(
        "SGLANG_TEST_ZIMAGE_MODEL_PATH",
        "Tongyi-MAI/Z-Image-Turbo",  # Supports both safetensors file and diffusers format
    )

    generator = DiffGenerator.from_pretrained(
        model_path=model_path,
        pipeline_class_name="ComfyUIZImagePipeline",
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
    req.negative_prompt_embeds = None
    req.raw_latent_shape = torch.tensor(latents.shape, dtype=torch.long)

    if req.guidance_scale > 1.0 and req.negative_prompt_embeds is not None:
        req.do_classifier_free_guidance = True
    else:
        req.do_classifier_free_guidance = False

    if req.seed is not None:
        generator_device = req.generator_device
        device_str = "cpu" if generator_device == "cpu" else "cuda"
        req.generator = [
            torch.Generator(device_str).manual_seed(req.seed + i)
            for i in range(req.num_outputs_per_prompt)
        ]
    else:
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]

    output_batch = generator._send_to_scheduler_and_wait_for_response([req])
    noise_pred = output_batch.noise_pred

    assert noise_pred is not None, "noise_pred should not be None in OutputBatch"
    assert isinstance(noise_pred, torch.Tensor), "noise_pred should be a torch.Tensor"
    assert (
        noise_pred.device.type == "cuda"
    ), f"noise_pred should be on cuda, got {noise_pred.device}"
    assert (
        noise_pred.dtype == torch.bfloat16
    ), f"noise_pred should be bfloat16, got {noise_pred.dtype}"

    print(f"✓ Successfully retrieved noise_pred from OutputBatch!")
    print(f"  noise_pred shape: {noise_pred.shape}")
    print(f"  noise_pred dtype: {noise_pred.dtype}")
    print(f"  noise_pred device: {noise_pred.device}")

    latents = output_batch.output if output_batch.output is not None else req.latents
    assert latents is not None, "latents should not be None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
