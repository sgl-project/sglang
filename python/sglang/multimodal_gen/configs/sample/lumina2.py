# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Lumina-Image-2.0 (T2I)."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class Lumina2SamplingParams(SamplingParams):
    """Defaults for Alpha-VLLM/Lumina-Image-2.0.

    Matches the diffusers Lumina2Pipeline defaults: guidance_scale=4.0 with 30
    FlowMatchEuler steps at 1024x1024.
    """

    data_type: DataType = DataType.IMAGE
    num_frames: int = 1
    guidance_scale: float = 4.0
    num_inference_steps: int = 30
    height: int = 1024
    width: int = 1024
    negative_prompt: str = ""
    # NOTE: off deliberately, and pinned so a change to the base default cannot
    # silently turn it on. Lumina's renorm-CFG is a different operation, and it
    # already runs in Lumina2PipelineConfig.postprocess_cfg_noise.
    cfg_normalization: float | bool = 0.0
