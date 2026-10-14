import math
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT = 16

VALID_BOT_TASKS = {
    "auto",
    "image",
    "think",
    "recaption",
    "think_recaption",
    "img_ratio",
    "none",
}

SYSTEM_PROMPT_PRESETS = {
    "none",
    "en_unified",
    "en_vanilla",
    "en_recaption",
    "en_think_recaption",
    "dynamic",
    "auto",
}


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    negative_prompt: str = ""
    num_frames: int = 1
    guidance_scale: float = 2.5
    num_inference_steps: int = 50

    # Tokenizer bot_task: controls the bot response prefix; "image" adds
    # no bot prefix in gen_image mode.
    bot_task: str = "image"

    # Preset name (see SYSTEM_PROMPT_PRESETS) or raw custom text.
    system_prompt: str | None = "en_unified"

    # Pre-generated CoT text from AR stage (think/recaption output)
    cot_text: str | None = None

    # Output geometry is separate from the generation canvas: the AR
    # processor picks its own bucket, then the decoder applies this
    # request-scoped policy to the decoded pixels.
    output_size_mode: str = "aspect_ratio"
    output_strategy: str = "native_crop"
    output_ratio_policy: str = "exact"
    output_crop_anchor: tuple[float, float] = (0.5, 0.5)
    output_max_ratio_error: float = 0.0005
    output_pad_value: float = 0.0

    # Supported resolutions (height, width) - must be divisible by 16
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1024, 1024),  # 1:1
            (768, 1024),  # 3:4 portrait
            (1024, 768),  # 4:3 landscape
            (720, 1280),  # 9:16 portrait
            (1280, 720),  # 16:9 landscape
        ]
    )

    def _adjust(self, server_args):
        # Pre-aligning each dimension to a 16-px grid here would distort the
        # processor's aspect-ratio bucket selection.
        if self.bot_task not in VALID_BOT_TASKS:
            logger.warning(
                f"Invalid bot_task '{self.bot_task}'. Must be one of {VALID_BOT_TASKS}. "
                f"Defaulting to 'image'."
            )
            self.bot_task = "image"
        self._validate_output_geometry()
        super()._adjust(server_args)

    def _validate_output_geometry(self) -> None:
        if self.output_size_mode not in {"aspect_ratio", "exact_size"}:
            raise ValueError("output_size_mode must be 'aspect_ratio' or 'exact_size'")
        if self.output_strategy not in {"native_crop", "native_pad"}:
            raise ValueError("output_strategy must be 'native_crop' or 'native_pad'")
        if self.output_ratio_policy not in {"exact", "approximate"}:
            raise ValueError("output_ratio_policy must be 'exact' or 'approximate'")
        if len(self.output_crop_anchor) != 2 or any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in self.output_crop_anchor
        ):
            raise ValueError(
                "output_crop_anchor must contain two finite values in [0, 1]"
            )
        if (
            not math.isfinite(self.output_max_ratio_error)
            or not 0.0 <= (self.output_max_ratio_error) < 1.0
        ):
            raise ValueError("output_max_ratio_error must be a finite value in [0, 1)")
        if (
            not math.isfinite(self.output_pad_value)
            or not 0.0 <= (self.output_pad_value) <= 1.0
        ):
            raise ValueError("output_pad_value must be a finite value in [0, 1]")


def align_hunyuan_image3_dimension(value: int) -> int:
    """Round a HunyuanImage-3 dimension up to a supported multiple."""
    return max(
        HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
        (value + HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT - 1)
        // HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT
        * HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
    )


def align_hunyuan_image3_resolution(width: int, height: int) -> tuple[int, int]:
    """Align both width and height to supported multiples."""
    return align_hunyuan_image3_dimension(width), align_hunyuan_image3_dimension(height)
