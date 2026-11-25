# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# VAE downsampling (8) times the DiT patch size (2)
_MOSS_IMAGE_SIZE_MULTIPLE = 16


def resolve_uncond_skip_layers(params: "MOSSImageSamplingParams") -> tuple[int, ...]:
    layers = params.uncond_skip_layers
    if layers is None:
        # The recommended two-pass mode is the default. Selecting three-pass
        # SLG keeps the unconditional branch complete unless explicitly overridden.
        return () if params.slg_layers and params.slg_scale > 0 else (22,)
    if not isinstance(layers, (list, tuple)):
        raise ValueError("uncond_skip_layers must be a list of indices.")
    return tuple(layers)


@dataclass
class MOSSImageSamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    seed: int | list[int] = field(default=0, metadata={"batch_sig_exclude": True})
    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    negative_prompt: str = ""
    # None selects the mode default; [] disables unconditional skipping.
    uncond_skip_layers: list[int] | None = None
    slg_layers: list[int] | None = field(default_factory=list)
    slg_scale: float = 0.0
    slg_start: float = 0.01
    slg_stop: float = 1.0

    def _adjust_visual_fields(self, server_args, pipeline_config) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        self._align_resolution(pipeline_config)

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Warmup overwrites the dimensions after _adjust has run.
        self._align_resolution(server_args.pipeline_config)

    def _align_resolution(self, pipeline_config) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.moss_image import (
            MOSSImagePipelineConfig,
        )

        # MOSSImageBeforeDenoisingStage rejects other sizes. The diffusers
        # backend keeps its own size handling.
        if not isinstance(pipeline_config, MOSSImagePipelineConfig):
            return
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            return
        if self.width <= 0 or self.height <= 0:
            return
        m = _MOSS_IMAGE_SIZE_MULTIPLE
        requested_size = (self.width, self.height)
        self.width = max(m, self.width // m * m)
        self.height = max(m, self.height // m * m)
        if (self.width, self.height) != requested_size:
            logger.warning(
                "MOSS-Image requires dimensions divisible by %s; adjusted requested "
                "resolution from %sx%s to %sx%s",
                m,
                *requested_size,
                self.width,
                self.height,
            )

    def should_use_classifier_free_guidance(self) -> bool:
        # The reference evaluates both branches even for guidance below one.
        if self.negative_prompt is None:
            self.negative_prompt = ""
        return self.guidance_scale != 1.0

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "uncond_skip_layers",
                "slg_layers",
                "slg_scale",
                "slg_start",
                "slg_stop",
            }
        )
