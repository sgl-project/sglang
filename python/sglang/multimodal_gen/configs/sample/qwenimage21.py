# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class QwenImage21SamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    num_frames: int = 1
    guidance_scale: float = 1.0
    num_inference_steps: int = 40
    negative_prompt: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._align_resolution()

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Warmup copies sampling defaults, then overwrites their dimensions
        # without invoking the dataclass constructor again.
        self._align_resolution()

    def _align_resolution(self) -> None:
        requested_size = (self.width, self.height)
        # Match Diffusers before request batching and pipeline validation
        # consume these dimensions.
        # Keep a positive sub-grid request usable, but leave invalid dimensions
        # to the existing request validation instead of rounding them positive.
        if self.width is not None and self.width > 0:
            self.width = max(32, self.width // 32 * 32)
        if self.height is not None and self.height > 0:
            self.height = max(32, self.height // 32 * 32)
        if (self.width, self.height) != requested_size:
            logger.warning(
                "Qwen-Image 2.1 requires dimensions divisible by 32; adjusted "
                "requested resolution from %sx%s to %sx%s",
                *requested_size,
                self.width,
                self.height,
            )

    @classmethod
    def default_image_output_format(cls) -> str:
        """Preserve Qwen-Image 2.1's RGBA outputs by default."""

        return "png"
