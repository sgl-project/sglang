# Krea-2 (K2) sampling defaults.
#
# `guidance_scale` is the SGLang classifier-free-guidance scale, which equals the
# K2 reference `cfg + 1` (SGLang combines `uncond + scale*(cond-uncond)`, the K2
# reference uses `cond + cfg*(cond-uncond)`). So K2 cfg=0 -> guidance_scale=1.0
# (no CFG), K2 cfg=3.5 -> guidance_scale=4.5.
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class Krea2SamplingParams(SamplingParams):
    """Distilled `oss_turbo` defaults: 8 steps, CFG disabled."""

    negative_prompt: str = ""
    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 1.0
    num_inference_steps: int = 8

    def _adjust_visual_fields(self, server_args, pipeline_config) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        self._align_resolution(pipeline_config)

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Warmup overwrites the dimensions after _adjust has run.
        self._align_resolution(server_args.pipeline_config)

    def _align_resolution(self, pipeline_config) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.krea2 import (
            Krea2PipelineConfig,
        )

        # Krea2BeforeDenoisingStage patchifies the latent with einops, which
        # fails on a latent side that is not a multiple of the patch size.
        if not isinstance(pipeline_config, Krea2PipelineConfig):
            return
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            return
        if self.width <= 0 or self.height <= 0:
            return
        m = (
            pipeline_config.get_vae_scale_factor()
            * pipeline_config.dit_config.arch_config.patch
        )
        requested_size = (self.width, self.height)
        self.width = max(m, self.width // m * m)
        self.height = max(m, self.height // m * m)
        if (self.width, self.height) != requested_size:
            logger.warning(
                "Krea-2 requires dimensions divisible by %s; adjusted requested "
                "resolution from %sx%s to %sx%s",
                m,
                *requested_size,
                self.width,
                self.height,
            )
