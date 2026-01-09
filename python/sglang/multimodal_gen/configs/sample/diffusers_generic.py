# SPDX-License-Identifier: Apache-2.0
"""
Generic sampling parameters for diffusers backend.

This module provides generic sampling parameters that work with any diffusers pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class DiffusersGenericSamplingParams(SamplingParams):
    """
    Generic sampling parameters for diffusers backend.

    These parameters cover the most common options across different diffusers pipelines.
    The diffusers pipeline will use whichever parameters it supports.

    For pipeline-specific parameters, use `diffusers_kwargs` dict which will be
    passed directly to the diffusers pipeline call.
    """

    # Override defaults with more conservative values that work across pipelines
    num_frames: int = 1  # default to image generation
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = ""

    # extra kwargs to pass directly to the diffusers pipeline
    # example: {"output_type": "latent", "return_dict": False}
    diffusers_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_frames > 1:
            self.data_type = DataType.VIDEO
        else:
            self.data_type = DataType.IMAGE

        if self.width is None:
            self.width_not_provided = True
            self.width = 1024
        if self.height is None:
            self.height_not_provided = True
            self.height = 1024
