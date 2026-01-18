from PIL import Image

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

RESOLUTION_PRESETS = {
    "16:9": {
        "480p": (832, 480),
        "580p": (960, 512),
        "720p": (1280, 720),
    },
    "9:16": {
        "480p": (480, 832),
        "580p": (512, 960),
        "720p": (720, 1280),
    },
    "1:1": {
        "480p": (480, 480),
        "580p": (512, 512),
        "720p": (720, 720),
    },
}

ASPECT_RATIO_MAP = {16 / 9: "16:9", 9 / 16: "9:16", 1.0: "1:1"}
ASPECT_RATIO_TOLERANCE = 0.01


def isotropic_crop_resize_pil(
    image: Image.Image, target_size: tuple[int, int]
) -> Image.Image:
    target_width, target_height = target_size
    orig_width, orig_height = image.size
    target_ratio = target_height / target_width
    orig_ratio = orig_height / orig_width

    if abs(orig_ratio - target_ratio) < ASPECT_RATIO_TOLERANCE:
        return image.resize((target_width, target_height), Image.LANCZOS)

    if orig_ratio > target_ratio:
        # Image is taller, crop height
        crop_height = int(target_ratio * orig_width)
        crop_width = orig_width
        y0 = (orig_height - crop_height) // 2
        y1 = y0 + crop_height
        x0, x1 = 0, orig_width
    else:
        # Image is wider, crop width
        crop_width = int(orig_height / target_ratio)
        crop_height = orig_height
        x0 = (orig_width - crop_width) // 2
        x1 = x0 + crop_width
        y0, y1 = 0, orig_height

    cropped_image = image.crop((x0, y0, x1, y1))
    resized_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)

    return resized_image


def parse_resolution_and_aspect_ratio(
    original_aspect_ratio: float, resolution: str | None, aspect_ratio: str | None
) -> tuple[int, int] | None:
    if resolution is None or aspect_ratio is None:
        return None

    if aspect_ratio == "auto":
        # according to the original aspect ratio of the image, select the closest resolution
        closest_aspect_ratio = min(
            ASPECT_RATIO_MAP.keys(), key=lambda x: abs(x - original_aspect_ratio)
        )
        aspect_ratio = ASPECT_RATIO_MAP[closest_aspect_ratio]
        logger.info(
            f"Based on the original aspect ratio {original_aspect_ratio}, the closest aspect ratio is {aspect_ratio}"
        )

    if aspect_ratio not in RESOLUTION_PRESETS:
        logger.warning(f"Invalid aspect_ratio: {aspect_ratio}, using default logic")
        return None

    if resolution not in RESOLUTION_PRESETS[aspect_ratio]:
        logger.warning(
            f"Resolution {resolution} not found for aspect_ratio {aspect_ratio}, using default logic"
        )
        return None

    return RESOLUTION_PRESETS[aspect_ratio][resolution]  # (width, height)
