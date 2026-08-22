# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class LLaDAImageSamplingParams(SamplingParams):
    data_type: DataType = DataType.IMAGE
    negative_prompt: str | None = None
    num_inference_steps: int = 20
    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 4.5
    max_sequence_length: int = 2048
