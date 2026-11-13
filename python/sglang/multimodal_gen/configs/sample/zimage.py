# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# VAE downsampling (8) times the DiT patch size (2)
_ZIMAGE_SIZE_MULTIPLE = 16


@dataclass
class ZImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50
    num_frames: int = 1
    negative_prompt: str = ""
    guidance_scale: float = 5.0
    cfg_normalization: float | bool = True

    def _adjust_visual_fields(self, server_args, pipeline_config) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        self._align_resolution(pipeline_config)

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Warmup overwrites the dimensions after _adjust has run.
        self._align_resolution(server_args.pipeline_config)

    def _align_resolution(self, pipeline_config) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
            ZImagePipelineConfig,
        )

        # The native DiT cannot split an odd latent side into 2x2 patches. The
        # diffusers backend keeps its own size handling.
        if not isinstance(pipeline_config, ZImagePipelineConfig):
            return
        if self.width is None and self.height is None:
            return
        requested_size = (self.width, self.height)
        # same fallback as InputValidationStage, so the derived side is aligned too
        if self.height is None:
            self.height = self.width * 720 // 1280
        elif self.width is None:
            self.width = self.height * 1280 // 720
        if self.width <= 0 or self.height <= 0:
            return
        m = _ZIMAGE_SIZE_MULTIPLE
        self.width = max(m, self.width // m * m)
        self.height = max(m, self.height // m * m)
        if (self.width, self.height) != requested_size:
            logger.warning(
                "Z-Image requires dimensions divisible by %s; adjusted requested "
                "resolution from %sx%s to %sx%s",
                m,
                *requested_size,
                self.width,
                self.height,
            )

    def should_use_classifier_free_guidance(self) -> bool:
        if self.negative_prompt is None:
            self.negative_prompt = ""
        return self.guidance_scale > 0.0


@dataclass
class ZImageTurboSamplingParams(ZImageSamplingParams):
    num_inference_steps: int = 9

    num_frames: int = 1
    # Z-Image officially recommends starting at 1024x1024
    height: int = 1024
    width: int = 1024
    # fps: int = 24

    guidance_scale: float = 0.0
    cfg_normalization: float | bool = False

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            coefficients=[
                7.33226126e02,
                -4.01131952e02,
                6.75869174e01,
                -3.14987800e00,
                9.61237896e-02,
            ],
        )
    )
