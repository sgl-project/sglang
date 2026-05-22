# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA image generation (T2I)."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SanaSamplingParams(SamplingParams):
    """Defaults for SANA 1.5 1024px variant.

    guidance_scale=4.5 enables standard classifier-free guidance.
    """

    data_type: DataType = DataType.IMAGE
    num_frames: int = 1
    guidance_scale: float = 4.5
    num_inference_steps: int = 20
    height: int = 1024
    width: int = 1024
    negative_prompt: str = (
        "low quality, low resolution, blurry, overexposed, underexposed, "
        "distorted, deformed, disfigured, bad anatomy, extra limbs, "
        "watermark, text, signature, ugly, noisy, artifacts"
    )
