from typing import Union

import numpy as np
import torch
from PIL import Image

DEFERRED_PREPROCESSING_KEY = "kimi_k3_deferred_preprocessing"


def prepare_kimi_k3_encoder_inputs(images, image_processor):
    """Keep K3 EPD images raw until the vision-DP owner is known.

    The lightweight NaViT shape calculation runs on every encoder rank.  The
    expensive resize, normalization, and patchification remain deferred in each
    item and are executed by ``KimiK3ForConditionalGeneration`` only for images
    assigned to the local vision rank.
    """
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput
    from sglang.srt.multimodal.processors.kimi_k25 import (
        _grid_thw_from_resize_config,
        navit_resize_config,
    )

    media_proc_cfg = getattr(image_processor, "media_proc_cfg", None)
    if not isinstance(media_proc_cfg, dict):
        raise ValueError(
            "Kimi-K3 EPD owner-side preprocessing requires "
            "image_processor.media_proc_cfg"
        )

    required = (
        "patch_size",
        "merge_kernel_size",
        "in_patch_limit",
        "patch_limit_on_one_side",
        "image_mean",
        "image_std",
    )
    missing = [name for name in required if name not in media_proc_cfg]
    if missing:
        raise ValueError(
            "Kimi-K3 image processor is missing deferred-preprocessing config: "
            + ", ".join(missing)
        )

    concrete_images = []
    for image in images:
        if isinstance(image, dict):
            if image.get("type") != "image" or "image" not in image:
                raise ValueError(f"Unsupported Kimi-K3 encoder media item: {image}")
            image = image["image"]
        concrete_images.append(image)

    patch_size = int(media_proc_cfg["patch_size"])
    merge_kernel_size = int(media_proc_cfg["merge_kernel_size"])
    common_deferred_config = {
        "image_mean": list(media_proc_cfg["image_mean"]),
        "image_std": list(media_proc_cfg["image_std"]),
        "transparent_bg_config": media_proc_cfg.get("transparent_bg_config"),
    }

    items = []
    grids = []
    original_image_sizes = []
    for image in concrete_images:
        width, height = (
            (int(image.shape[-1]), int(image.shape[-2]))
            if isinstance(image, torch.Tensor)
            else image.size
        )
        resize_config = navit_resize_config(
            width,
            height,
            patch_size,
            merge_kernel_size,
            int(media_proc_cfg["in_patch_limit"]),
            int(media_proc_cfg["patch_limit_on_one_side"]),
            media_proc_cfg.get("fixed_output_tokens"),
        )
        grid_thw = _grid_thw_from_resize_config(resize_config, patch_size)
        grid_tensor = torch.tensor([grid_thw], dtype=torch.int64)
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=to_chw_uint8(image),
                model_specific_data={
                    "grid_thws": grid_tensor,
                    DEFERRED_PREPROCESSING_KEY: {
                        **common_deferred_config,
                        "resize_config": resize_config,
                    },
                },
            )
        )
        grids.append(grid_thw)
        original_image_sizes.append([width, height])

    grid_thws = torch.tensor(grids, dtype=torch.int64)
    return EncoderPreprocessOutput(
        {
            # Preserve the conventional feature key for cache/accounting code;
            # EncoderPreprocessOutput.mm_items is the authoritative per-item view.
            "pixel_values": [item.feature for item in items],
            "grid_thws": grid_thws,
            "original_image_sizes": original_image_sizes,
        },
        mm_items=items,
    )


def to_chw_uint8(
    image: Union[torch.Tensor, Image.Image],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(image, Image.Image):
        has_alpha = image.mode != "RGB" and (
            "A" in image.getbands() or "transparency" in image.info
        )
        array = np.array(image.convert("RGBA" if has_alpha else "RGB"), copy=True)
        image = torch.from_numpy(array).permute(2, 0, 1)

    if image.dtype != torch.uint8:
        raise ValueError(
            f"Kimi-K3 preprocessing expects raw uint8 pixels, got {image.dtype}"
        )
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if device is not None:
        image = image.to(device)
    return image


def fill_transparent_bg(x: torch.Tensor, bg_config: Union[dict, None]) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    rgb = x[:, :3]
    if bg_config is None:
        return rgb

    _, _, height, width = x.shape
    pattern = bg_config.get("pattern", "black")
    if pattern == "chessboard":
        square = bg_config.get("chessboard_square_size", 16)
        white = float(bg_config.get("chessboard_white_value", 255))
        gray = float(bg_config.get("chessboard_gray_value", 200))
        top_left = bg_config.get("chessboard_square_on_top_left", True)
        ys = torch.arange(height, device=x.device) // square
        xs = torch.arange(width, device=x.device) // square
        parity = (ys.unsqueeze(1) + xs.unsqueeze(0)) % 2
        background = torch.where(parity == (1 if top_left else 0), gray, white)
        background = background.unsqueeze(0).expand(3, height, width)
    elif pattern == "white":
        background = torch.full((3, height, width), 255.0, device=x.device)
    elif pattern == "black":
        background = torch.zeros(3, height, width, device=x.device)
    elif pattern == "gray":
        background = torch.full((3, height, width), 128.0, device=x.device)
    else:
        raise ValueError(f"Invalid background pattern: {pattern}")

    alpha = (x[:, 3:4] / 255.0).clamp(0.0, 1.0)
    return (alpha * rgb + (1.0 - alpha) * background).clamp(0.0, 255.0).floor_()


def normalization_tensors(
    image_mean: list[float],
    image_std: list[float],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.tensor(
        [1.0 / (255.0 * std) for std in image_std],
        device=device,
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    bias = torch.tensor(
        [-mean / std for mean, std in zip(image_mean, image_std)],
        device=device,
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    return scale, bias
