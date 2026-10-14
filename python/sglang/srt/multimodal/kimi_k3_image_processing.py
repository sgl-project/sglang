import functools
import math
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

DEFERRED_PREPROCESSING_KEY = "kimi_k3_deferred_preprocessing"


@dataclass(frozen=True)
class KimiK3DeferredPreprocessing:
    """Parameters the vision-DP owner needs to finish one deferred image.

    ``backend`` is the producer's decision and cannot be recovered from the
    item; the feature's own layout stays observable on ``item.feature``, so it
    is not mirrored here.
    """

    backend: Literal["gpu", "cpu"]
    image_mean: list[float]
    image_std: list[float]
    transparent_bg_config: Optional[dict]
    resize_config: dict


def prepare_kimi_k3_encoder_inputs(
    images, image_processor, *, use_gpu_preprocessing=False
):
    """Keep K3 EPD images raw until the vision-DP owner is known.

    The lightweight NaViT shape calculation runs on every encoder rank.  The
    expensive resize, normalization, and patchification remain deferred in each
    item and are executed by ``KimiK3ForConditionalGeneration`` only for images
    assigned to the local vision rank.
    """
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.multimodal.encoder_preprocessing import (
        EncoderPreprocessOutput,
        hash_raw_encoder_item,
    )
    from sglang.srt.multimodal.processors.kimi_k25 import (
        _grid_thw_from_resize_config,
        navit_resize_config,
    )

    try:
        media_proc_cfg = image_processor.media_proc_cfg
    except AttributeError as exc:
        raise ValueError(
            "Kimi-K3 EPD owner-side preprocessing requires "
            "image_processor.media_proc_cfg"
        ) from exc
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
    content_digests = []
    for image in images:
        content_digest = None
        if isinstance(image, dict):
            if image.get("type") != "image" or "image" not in image:
                raise ValueError(f"Unsupported Kimi-K3 encoder media item: {image}")
            content_digest = image.get("content_hash")
            image = image["image"]
        concrete_images.append(image)
        content_digests.append(content_digest)

    patch_size = int(media_proc_cfg["patch_size"])
    merge_kernel_size = int(media_proc_cfg["merge_kernel_size"])
    deferred_preprocessing = functools.partial(
        KimiK3DeferredPreprocessing,
        backend="gpu" if use_gpu_preprocessing else "cpu",
        image_mean=list(media_proc_cfg["image_mean"]),
        image_std=list(media_proc_cfg["image_std"]),
        transparent_bg_config=media_proc_cfg.get("transparent_bg_config"),
    )

    items = []
    grids = []
    original_image_sizes = []
    for image, content_digest in zip(concrete_images, content_digests):
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
        model_specific_data = {
            "grid_thws": grid_tensor,
            DEFERRED_PREPROCESSING_KEY: deferred_preprocessing(
                resize_config=resize_config
            ),
        }
        if content_digest is not None:
            model_specific_data["content_digest"] = content_digest
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=to_chw_uint8(image) if use_gpu_preprocessing else image,
            model_specific_data=model_specific_data,
        )
        if not use_gpu_preprocessing:
            item.set_hash(hash_raw_encoder_item(image))
        items.append(item)
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
        item_sizes=[math.prod(grid) for grid in grids],
        materialize_local_items=(
            None
            if use_gpu_preprocessing
            else functools.partial(
                materialize_kimi_k3_cpu_item_features,
                image_processor=image_processor,
            )
        ),
    )


def materialize_kimi_k3_cpu_features(items, image_processor) -> torch.Tensor:
    """Run the checkpoint's exact processor only on locally owned images."""
    medias = []
    for item in items:
        image = item.feature
        if not isinstance(image, Image.Image):
            if not isinstance(image, torch.Tensor) or image.dtype != torch.uint8:
                raise TypeError(
                    "Kimi-K3 deferred CPU preprocessing expects PIL or uint8 tensors"
                )
            image = to_hwc_uint8(image).numpy()
            channels = image.shape[-1]
            if channels == 1:
                image = Image.fromarray(image[..., 0], mode="L")
            elif channels == 3:
                image = Image.fromarray(image, mode="RGB")
            elif channels == 4:
                image = Image.fromarray(image, mode="RGBA")
            else:
                raise ValueError(f"Unsupported Kimi-K3 image channel count: {channels}")
        medias.append({"type": "image", "image": image})

    output = image_processor.preprocess(medias, return_tensors="pt")
    expected_grids = torch.cat(
        [item.model_specific_data["grid_thws"] for item in items], dim=0
    )
    if not torch.equal(output["grid_thws"].cpu(), expected_grids.cpu()):
        raise ValueError("Kimi-K3 deferred CPU preprocessing produced wrong grids")
    return output["pixel_values"]


def materialize_kimi_k3_cpu_item_features(items, image_processor) -> list[torch.Tensor]:
    """Return exact checkpoint-processor features split by logical image."""
    pixel_values = materialize_kimi_k3_cpu_features(items, image_processor)
    patch_counts = [
        math.prod(item.model_specific_data["grid_thws"][0].tolist()) for item in items
    ]
    if sum(patch_counts) != pixel_values.shape[0]:
        raise ValueError(
            "Kimi-K3 processor feature length does not match image grids: "
            f"{pixel_values.shape[0]} != {sum(patch_counts)}"
        )
    return list(pixel_values.split(patch_counts))


def to_hwc_uint8(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    """Stage an exact CPU image without doing resize/normalize work."""
    if isinstance(image, Image.Image):
        has_alpha = image.mode != "RGB" and (
            "A" in image.getbands() or "transparency" in image.info
        )
        array = np.array(image.convert("RGBA" if has_alpha else "RGB"), copy=True)
        return torch.from_numpy(array)

    if image.dtype != torch.uint8:
        raise ValueError(
            f"Kimi-K3 preprocessing expects raw uint8 pixels, got {image.dtype}"
        )
    if image.dim() == 2:
        image = image.unsqueeze(-1)
    elif image.dim() == 3 and image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    return image.cpu().contiguous()


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
