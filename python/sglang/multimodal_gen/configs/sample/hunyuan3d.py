# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Hunyuan3D generation."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class Hunyuan3DSamplingParams(SamplingParams):
    """Sampling parameters for Hunyuan3D image-to-mesh generation."""

    seed: int = 0
    negative_prompt: str = ""

    def __post_init__(self):
        if self.prompt is None:
            self.prompt = ""

        super().__post_init__()
