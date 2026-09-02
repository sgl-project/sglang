# copy from https://huggingface.co/OpenGVLab/InternVL3-1B
import math

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    input_size,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def image_to_pixel_values(
    image: Image.Image,
    *,
    input_size: int,
    min_num_tiles: int = 1,
    max_num_tiles: int,
    use_thumbnail: bool,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    images = dynamic_preprocess(
        image,
        min_num=min_num_tiles,
        max_num=max_num_tiles,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )
    transform = build_transform(input_size, mean=mean, std=std)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def compute_dynamic_image_size(
    orig_w: int,
    orig_h: int,
    patch_size: int,
    downsample_ratio: float,
    min_num_patches: int,
    max_num_patches: int,
) -> tuple[int, int, int]:
    """Compute optimal resize dimensions for dynamic resolution.

    The image is resized (not tiled) to a variable size that respects the
    aspect ratio while staying within the patch budget. Dimensions are
    snapped to multiples of ``patch_size * ds`` so that pixel-shuffle
    downsampling produces integer grid sizes.

    Returns:
        (target_w, target_h, num_tokens) where num_tokens is the
        post-pixel-shuffle token count.
    """
    ds = int(1 / downsample_ratio)
    snap = patch_size * ds

    pw = max(1, round(orig_w / patch_size))
    ph = max(1, round(orig_h / patch_size))
    native_patches = pw * ph

    budget = min(native_patches, max_num_patches)
    budget = max(budget, min_num_patches)
    factor = math.sqrt(budget / max(native_patches, 1))
    factor = min(factor, 1.0)

    target_pw = max(ds, int(round(pw * factor / ds)) * ds)
    target_ph = max(ds, int(round(ph * factor / ds)) * ds)

    if target_pw * target_ph < min_num_patches:
        up = math.sqrt(min_num_patches / (target_pw * target_ph))
        target_pw = max(ds, int(math.ceil(target_pw * up / ds)) * ds)
        target_ph = max(ds, int(math.ceil(target_ph * up / ds)) * ds)

    if target_pw * target_ph > max_num_patches:
        down = math.sqrt(max_num_patches / (target_pw * target_ph))
        target_pw = max(ds, int(math.floor(target_pw * down / ds)) * ds)
        target_ph = max(ds, int(math.floor(target_ph * down / ds)) * ds)

    target_w = target_pw * patch_size
    target_h = target_ph * patch_size
    num_tokens = (target_pw * target_ph) // (ds * ds)

    return target_w, target_h, num_tokens


def dynamic_resize_image(
    image: Image.Image,
    patch_size: int,
    downsample_ratio: float,
    min_num_patches: int,
    max_num_patches: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[torch.Tensor, int]:
    """Resize image for dynamic resolution and return pixel tensor + token count.

    Returns:
        (pixel_values [1, 3, H, W], num_tokens)
    """
    orig_w, orig_h = image.size
    target_w, target_h, num_tokens = compute_dynamic_image_size(
        orig_w,
        orig_h,
        patch_size,
        downsample_ratio,
        min_num_patches,
        max_num_patches,
    )
    image = image.convert("RGB")
    image = image.resize((target_w, target_h), Image.BICUBIC)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values, num_tokens


def resize_image_to_pixels(
    image: Image.Image,
    target_w: int,
    target_h: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Resize image to exact target dimensions and return normalized tensor.

    Returns:
        pixel_values tensor of shape [1, 3, target_h, target_w].
    """
    image = image.convert("RGB")
    image = image.resize((target_w, target_h), Image.BICUBIC)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform(image).unsqueeze(0)


def compute_budgeted_image_sizes(
    image_sizes: list[tuple[int, int]],
    total_token_budget: int,
    patch_size: int,
    downsample_ratio: float,
    min_num_patches: int,
    max_num_patches: int,
    max_iterations: int = 10,
) -> list[tuple[int, int, int]]:
    """Compute per-image sizes that fit within a total token budget.

    When multiple images share a prompt, their combined post-pixel-shuffle
    tokens must not exceed ``total_token_budget``.  This function iteratively
    reduces per-image patch limits until the total fits.

    Returns:
        List of (target_w, target_h, num_tokens) per image.
    """
    n = len(image_sizes)
    if n == 0:
        return []

    ds = int(round(1 / downsample_ratio))
    per_image_max = [max_num_patches] * n
    results: list[tuple[int, int, int]] = []

    for _ in range(max_iterations):
        results = [
            compute_dynamic_image_size(
                orig_w,
                orig_h,
                patch_size,
                downsample_ratio,
                min_num_patches,
                per_image_max[i],
            )
            for i, (orig_w, orig_h) in enumerate(image_sizes)
        ]
        total_tokens = sum(num_tokens for _, _, num_tokens in results)

        if total_tokens <= total_token_budget:
            return results

        scale = total_token_budget / total_tokens
        for i in range(n):
            current_patches = results[i][2] * ds * ds
            per_image_max[i] = max(min_num_patches, int(current_patches * scale))

    return results


def get_video_target_size_and_feature_size(
    orig_w: int,
    orig_h: int,
    target_num_patches: int,
    maintain_aspect_ratio: bool,
    patch_size: int,
    downsample_ratio: float,
) -> tuple[int, int, int]:
    """Compute target resize dimensions and post-downsample token count for video.

    Single source of truth for video spatial dimensions — used by both
    video_to_pixel_values (resize) and the processor (token counting).

    Returns:
        (target_w, target_h, feature_size) where feature_size is the
        post-pixel-shuffle token count.
    """
    ds = int(1 / downsample_ratio)

    if target_num_patches > 0 and maintain_aspect_ratio:
        aspect = orig_w / max(orig_h, 1)
        ph = math.sqrt(target_num_patches / max(aspect, 1e-6))
        pw = ph * aspect
        target_pw = max(ds, int(round(pw / ds)) * ds)
        target_ph = max(ds, int(round(ph / ds)) * ds)
    elif target_num_patches > 0:
        side = int(math.sqrt(target_num_patches))
        target_pw = max(ds, int(round(side / ds)) * ds)
        target_ph = target_pw
    else:
        target_pw = max(ds, round(orig_w / patch_size / ds) * ds)
        target_ph = max(ds, round(orig_h / patch_size / ds) * ds)

    target_w = target_pw * patch_size
    target_h = target_ph * patch_size
    feature_size = (target_pw // ds) * (target_ph // ds)

    return target_w, target_h, feature_size


def video_to_pixel_values(
    frame: Image.Image,
    patch_size: int,
    downsample_ratio: float,
    target_num_patches: int,
    maintain_aspect_ratio: bool,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[torch.Tensor, int]:
    """Resize a single video frame for temporal compression pipeline.

    Returns:
        (pixel_values [1, 3, H, W], feature_size) where feature_size is
        the post-pixel-shuffle token count.
    """
    orig_w, orig_h = frame.size
    target_w, target_h, feature_size = get_video_target_size_and_feature_size(
        orig_w,
        orig_h,
        target_num_patches,
        maintain_aspect_ratio,
        patch_size,
        downsample_ratio,
    )

    frame = frame.convert("RGB")
    frame = frame.resize((target_w, target_h), Image.BICUBIC)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    pixel_values = transform(frame).unsqueeze(0)
    return pixel_values, feature_size
