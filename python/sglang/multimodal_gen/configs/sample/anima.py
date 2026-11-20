# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

# VAE downsampling (8) times the DiT patch size (2)
_ANIMA_SIZE_MULTIPLE = 16


@dataclass
class AnimaSamplingParams(SamplingParams):
    _request_size_model_name = "Anima"

    height: int = 1024
    width: int = 1024
    num_frames: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    negative_prompt: str = ""

    def _request_size_multiple(self, pipeline_config) -> int | None:
        from sglang.multimodal_gen.configs.pipeline_configs.anima import (
            AnimaPipelineConfig,
        )

        # The native DiT rejects an odd latent side, which it cannot split into
        # 2x2 patches. The diffusers backend keeps its own size handling.
        if not isinstance(pipeline_config, AnimaPipelineConfig):
            return None
        return _ANIMA_SIZE_MULTIPLE
