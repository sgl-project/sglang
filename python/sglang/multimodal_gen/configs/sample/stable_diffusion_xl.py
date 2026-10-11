# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Stable Diffusion XL text-to-image generation."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class StableDiffusionXLSamplingParams(SamplingParams):
    """Defaults match Diffusers' StableDiffusionXLPipeline.__call__."""

    data_type: DataType = DataType.IMAGE
    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    # Empty means zero negative embeddings (force_zeros_for_empty_prompt), not "" encoded.
    negative_prompt: str = ""
