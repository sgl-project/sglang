"""Image preprocessing.

An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and a `n_llm_h x n_llm_w` token grid
after the 3x3 aligner downsample, which the LLM sees as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every one of those positions carries `image_token_id` in `input_ids`; only the token type tells them
apart. The IMAGE slots are filled with aligner rows in reading order.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


GPU_PLAN_KEY = "dsv41_gpu_plan"


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height
    )
    return math.floor(height * beta / patch_size) * patch_size, math.floor(
        width * beta / patch_size
    ) * patch_size


def safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, args):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = args.vision_patch_size
    if (
        args.vision_max_wh_ratio is not None
        and width > height * args.vision_max_wh_ratio
    ):
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        args.vision_downsample_ratio,
        args.vision_max_n_token,
    )


def to_rgb(image: Image.Image) -> Image.Image:
    """The same RGB conversion for every preprocessing backend."""
    return image.convert("RGB")


def patchify_image(image, args):
    p = args.vision_patch_size
    image = to_rgb(image)
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
        image.width, image.height, args
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if (
        args.vision_max_wh_ratio is not None
        and image.width >= args.vision_max_wh_ratio * image.height
    ):
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


def prepare_image(image, args):
    image = to_rgb(image)
    lh, lw, height, width = plan_image_grid(image.width, image.height, args)
    stretch = (
        args.vision_max_wh_ratio is not None
        and image.width >= args.vision_max_wh_ratio * image.height
    )
    resize_h, resize_w = height, width
    if not stretch:
        if image.width / image.height > width / height:
            resize_h = round(image.height / image.width * width)
        elif image.width / image.height < width / height:
            resize_w = round(image.width / image.height * height)
    plan = {
        "height": height,
        "width": width,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "top": round((height - resize_h) / 2),
        "left": round((width - resize_w) / 2),
        "patch_size": args.vision_patch_size,
    }
    return np.array(image, dtype=np.uint8), plan, lh, lw


def patchify_image_rust(image, args, *, resize_patchify):
    pixels, plan, lh, lw = prepare_image(image, args)
    bits = resize_patchify(
        pixels,
        (plan["height"], plan["width"]),
        (plan["resize_h"], plan["resize_w"]),
        (plan["top"], plan["left"]),
        plan["patch_size"],
    )
    p = plan["patch_size"]
    h, w = plan["height"] // p, plan["width"] // p
    patches = torch.from_numpy(bits).view(torch.bfloat16).view(h * w, 3, p, p)
    return patches, h, w, lh, lw


def prepare_image_gpu(image, args):
    pixels, plan, lh, lw = prepare_image(image, args)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous(), plan, lh, lw


def materialize_image_gpu(pixels: torch.Tensor, plan: dict) -> torch.Tensor:
    """Resize, pad, normalize and patchify on the input tensor's device."""
    x = pixels.unsqueeze(0).float()
    target = (plan["resize_h"], plan["resize_w"])
    # PIL resizes separably, rounding and clamping to uint8 after each pass;
    # fusing the two passes into one float resize diverges on high-contrast images.
    for size in ((x.shape[-2], target[1]), target):
        if x.shape[-2:] != size:
            x = (
                F.interpolate(
                    x, size=size, mode="bicubic", align_corners=False, antialias=True
                )
                .round()
                .clamp_(0, 255)
            )
    top, left = plan["top"], plan["left"]
    x = F.pad(
        x,
        (left, plan["width"] - target[1] - left, top, plan["height"] - target[0] - top),
        value=127,
    )
    x = ((x / 255 - 0.5) / 0.5).to(torch.bfloat16)
    p = plan["patch_size"]
    h, w = plan["height"] // p, plan["width"] // p
    return x.reshape(3, h, p, w, p).permute(1, 3, 0, 2, 4).reshape(h * w, 3, p, p)
