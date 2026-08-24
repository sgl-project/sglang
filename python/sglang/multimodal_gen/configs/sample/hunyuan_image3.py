from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT = 16

# Valid bot_task values for the tokenizer
VALID_BOT_TASKS = {"auto", "image", "think", "recaption", "img_ratio", "none"}

# Valid sys_type values for system prompt selection
VALID_SYS_TYPES = {
    "none",           # No system prompt
    "en_unified",     # Unified English system prompt (default for HunyuanImage-3)
    "en_vanilla",     # Vanilla English system prompt
    "en_recaption",   # Recaption English system prompt
    "en_think_recaption",  # Think + recaption English system prompt
    "auto",           # Auto-select based on bot_task
}


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    """Sampling parameters for HunyuanImage-3."""

    negative_prompt: str = ""
    num_frames: int = 1
    guidance_scale: float = 2.5
    num_inference_steps: int = 50

    # HunyuanImage-3 specific params
    # Mode: auto, image, recaption, think, img_ratio
    mode: str = "auto"

    # Tokenizer bot_task: controls the bot response prefix in the tokenizer.
    # Options: auto, image, think, recaption, img_ratio, none
    # For image generation, "image" or "none" are typical choices.
    # Default: "image" (no bot prefix added for gen_image mode)
    bot_task: str = "image"

    # System prompt type: controls which system prompt to use.
    # Options: none, en_unified, en_vanilla, en_recaption, en_think_recaption, auto
    # Default: "en_unified" (matches generation_config.json)
    sys_type: str = "en_unified"

    # CoT (Chain-of-Thought) related
    enable_cot: bool = False
    cot_mode: str = "recaption"  # recaption or think

    # Image size control
    image_size: str = "1024x1024"

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
        requested_width = self.width
        requested_height = self.height
        if self.width is not None and self.height is not None:
            self.width, self.height = align_hunyuan_image3_resolution(
                self.width, self.height
            )
            if (self.width, self.height) != (
                requested_width,
                requested_height,
            ):
                logger.warning(
                    "HunyuanImage-3 requires dimensions divisible by %s; adjusted "
                    "requested resolution from %sx%s to %sx%s",
                    HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
                    requested_width,
                    requested_height,
                    self.width,
                    self.height,
                )
        # Validate bot_task and sys_type
        if self.bot_task not in VALID_BOT_TASKS:
            logger.warning(
                f"Invalid bot_task '{self.bot_task}'. Must be one of {VALID_BOT_TASKS}. "
                f"Defaulting to 'image'."
            )
            self.bot_task = "image"
        if self.sys_type not in VALID_SYS_TYPES:
            logger.warning(
                f"Invalid sys_type '{self.sys_type}'. Must be one of {VALID_SYS_TYPES}. "
                f"Defaulting to 'en_unified'."
            )
            self.sys_type = "en_unified"
        super()._adjust(server_args)


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
