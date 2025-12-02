# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class QwenImageSamplingParams(SamplingParams):
    # Video parameters
    # height: int = 1024
    # width: int = 1024
    negative_prompt: str = " "
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 4.0
    num_inference_steps: int = 50


@dataclass
class QwenImageEditPlusSamplingParams(QwenImageSamplingParams):
    # Denoising stage
    guidance_scale: float = 1.0
    # "true_cfg_scale": 4.0 TODO(yhyang201): check if this is correct
    num_inference_steps: int = 40
