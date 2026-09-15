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
    num_inference_steps: int = 4
    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 1.0
    max_sequence_length: int = 2048

    def _validate_with_pipeline_config(self, pipeline_config):
        super()._validate_with_pipeline_config(pipeline_config)
        if self.enable_cache_dit or self.cache_dit_params is not None:
            raise ValueError("LLaDA-Image does not support cache-dit")
        if self.max_sequence_length is not None and not (
            0 < self.max_sequence_length <= pipeline_config.max_request_text_tokens
        ):
            raise ValueError(
                "LLaDA-Image max_sequence_length must be between 1 and 3584"
            )
