from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

GLM_IMAGE_RESOLUTION_ALIGNMENT = 32


@dataclass
class GlmImageSamplingParams(SamplingParams):
    negative_prompt = ""

    num_frames: int = 1
    guidance_scale: float = 1.5
    num_inference_steps: int = 30

    # Preserve the user-facing canvas before width/height are expanded to the
    # D32 generation grid. These fields intentionally participate in dynamic
    # batch compatibility because every item in a decoded tensor batch must use
    # the same crop.
    requested_width: int | None = field(default=None, init=False)
    requested_height: int | None = field(default=None, init=False)

    def _adjust(self, server_args):
        requested_width = self.width
        requested_height = self.height
        if self.width is not None and self.height is not None:
            if self.requested_width is None:
                self.requested_width = requested_width
            if self.requested_height is None:
                self.requested_height = requested_height
            self.width, self.height = align_glm_image_resolution(
                self.width, self.height
            )
            if (self.width, self.height) != (
                requested_width,
                requested_height,
            ):
                logger.warning(
                    "GLM-Image requires dimensions divisible by %s; adjusted "
                    "requested resolution from %sx%s to %sx%s",
                    GLM_IMAGE_RESOLUTION_ALIGNMENT,
                    requested_width,
                    requested_height,
                    self.width,
                    self.height,
                )
        super()._adjust(server_args)


def align_glm_image_dimension(value: int) -> int:
    """Round a GLM-Image dimension up to a supported multiple."""
    return max(
        GLM_IMAGE_RESOLUTION_ALIGNMENT,
        (value + GLM_IMAGE_RESOLUTION_ALIGNMENT - 1)
        // GLM_IMAGE_RESOLUTION_ALIGNMENT
        * GLM_IMAGE_RESOLUTION_ALIGNMENT,
    )


def align_glm_image_resolution(width: int, height: int) -> tuple[int, int]:
    return align_glm_image_dimension(width), align_glm_image_dimension(height)
