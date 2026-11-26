# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# VAE downsampling (8) times the DiT patch size (2)
_ANIMA_SIZE_MULTIPLE = 16


@dataclass
class AnimaSamplingParams(SamplingParams):
    height: int = 1024
    width: int = 1024
    num_frames: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    negative_prompt: str = ""

    def _adjust_visual_fields(self, server_args, pipeline_config) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        self._align_resolution(pipeline_config)

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Warmup overwrites the dimensions after _adjust has run.
        self._align_resolution(server_args.pipeline_config)

    def _align_resolution(self, pipeline_config) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.anima import (
            AnimaPipelineConfig,
        )

        # The native DiT rejects an odd latent side, which it cannot split into
        # 2x2 patches. The diffusers backend keeps its own size handling.
        if not isinstance(pipeline_config, AnimaPipelineConfig):
            return
        if self.width is None or self.height is None:
            return
        if self.width <= 0 or self.height <= 0:
            return
        multiple = _ANIMA_SIZE_MULTIPLE
        requested_size = (self.width, self.height)
        self.width = max(multiple, self.width // multiple * multiple)
        self.height = max(multiple, self.height // multiple * multiple)
        if (self.width, self.height) != requested_size:
            logger.warning(
                "Anima requires dimensions divisible by %s; adjusted requested "
                "resolution from %sx%s to %sx%s",
                multiple,
                *requested_size,
                self.width,
                self.height,
            )
