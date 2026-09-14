# Modified for SGLang; see this directory's README.md for upstream source.

from __future__ import annotations

import math

import torch
import torchvision.transforms as T
from PIL import Image

SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to `number` that is divisible by `factor`."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer >= `number` that is divisible by `factor`."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer <= `number` that is divisible by `factor`."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
) -> tuple[int, int]:
    """Rescale so that H/W are divisible by `factor` and total pixels ∈ [min, max].

    Copied from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L60
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def dynamic_preprocess_native_resolution(
    image: Image.Image,
    size_factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    **_kwargs,
) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


def preprocess_pixel_values(pixel_values: torch.Tensor, patch_size: int = 16):
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size

    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
        .reshape(grid_h * grid_w, c * patch_size**2)
    )

    grid_hw = torch.tensor([[grid_h, grid_w]], device=pixel_values.device)
    return flatten_pixel_values, grid_hw


def get_contrasting_background(image: Image.Image):
    """Return a background color for RGBA->RGB conversion, or ``None`` to use default.

    The original Neo_Unify implementation computed a contrasting background
    from the alpha channel. For this open-source release we fall back to a
    plain white background; callers that need the smarter behavior can override
    this function.
    """
    del image
    return (255, 255, 255)


def load_image_native(
    image,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
):
    """Load and preprocess an image: RGB convert, smart-resize, normalize, patchify."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGBA":
        bg_color = get_contrasting_background(image)
        if bg_color:
            background = Image.new("RGB", image.size, bg_color)
            background.paste(image, mask=image.split()[3])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    new_image = dynamic_preprocess_native_resolution(
        image,
        size_factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    pixel_values, grid_hw = preprocess_pixel_values(
        transform(new_image).to(torch.float32), patch_size=patch_size
    )
    return pixel_values, grid_hw
