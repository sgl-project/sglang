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
    # Base checkpoint recipe from the model card.
    num_inference_steps: int = 50
    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 5.0
    max_sequence_length: int = 2048

    def _adjust(self, server_args):
        super()._adjust(server_args)
        # The offline generator pairs prompts from a list or a file with one
        # image each, so only a single string prompt is checked here.
        if (
            isinstance(self.prompt, str)
            and self.prompt_path is None
            and getattr(server_args, "prompt_file_path", None) is None
            and isinstance(self.image_path, list)
            and len(self.image_path) > 1
        ):
            raise ValueError(
                "LLaDA-Image editing takes exactly one source image, "
                f"got {len(self.image_path)}"
            )

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
        pipeline_config.validate_output_size(
            self.width, self.height, editing=bool(self.image_path)
        )


@dataclass
class LLaDAImageTurboSamplingParams(LLaDAImageSamplingParams):
    # The distilled Turbo checkpoint samples in 4 steps without CFG.
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
