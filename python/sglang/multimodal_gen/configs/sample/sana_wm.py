# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-WM TI2V world model generation."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SanaWMSamplingParams(SamplingParams):
    """
    Default sampling parameters for SANA-WM 720p video generation.

    SANA-WM generates 720p video (704×1280) at 16fps.
    Supported frame counts: must satisfy (num_frames - 1) % 8 == 0.
    Common choices: 49 (~3s), 81 (~5s), 121 (~7.5s), 321 (~20s), 961 (~60s).
    """

    data_type: DataType = DataType.VIDEO

    # Resolution: 720p landscape (LTX-2 VAE requires multiples of 32)
    height: int = 704
    width: int = 1280

    # Frame count: 49 = (49-1)/8 = 6 latent frames → ~3 seconds at 16fps
    num_frames: int = 49

    # SANA-WM inference steps: 20 steps is a good default for quality/speed
    num_inference_steps: int = 20

    # Classifier-free guidance scale
    guidance_scale: float = 4.5

    negative_prompt: str = (
        "low quality, low resolution, blurry, overexposed, underexposed, "
        "distorted, deformed, disfigured, bad anatomy, extra limbs, "
        "watermark, text, signature, ugly, noisy, artifacts, camera shake, "
        "jitter, lens flare"
    )
