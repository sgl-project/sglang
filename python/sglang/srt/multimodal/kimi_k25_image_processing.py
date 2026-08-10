import functools
import math

import numpy as np
import torch
from PIL import Image

from sglang.srt.multimodal.cpu_image_processing import (
    materialize_exact_navit_image_features,
    supports_exact_navit_cpu_preprocessing,
)


def _to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, torch.Tensor) or image.dtype != torch.uint8:
        raise TypeError("Kimi-K2.5 encoder preprocessing expects PIL or uint8 tensors")

    array = image.detach().cpu()
    if array.ndim == 2:
        array = array.unsqueeze(-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = array.permute(1, 2, 0)
    array = array.contiguous().numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def _kimi_k25_cpu_medias(items):
    return [{"type": "image", "image": _to_pil_rgb(item.feature)} for item in items]


def _materialize_kimi_k25_cpu_items_reference(items, image_processor):
    output = image_processor.preprocess(
        _kimi_k25_cpu_medias(items), return_tensors="pt"
    )
    patch_counts = [
        math.prod(item.model_specific_data["grid_thws"][0].tolist()) for item in items
    ]
    if sum(patch_counts) != output["pixel_values"].shape[0]:
        raise ValueError(
            "Kimi-K2.5 processor feature length does not match image grids: "
            f"{output['pixel_values'].shape[0]} != {sum(patch_counts)}"
        )
    return list(output["pixel_values"].split(patch_counts)), output["grid_thws"]


def materialize_kimi_k25_cpu_item_features(
    items, image_processor
) -> list[torch.Tensor]:
    medias = _kimi_k25_cpu_medias(items)
    if supports_exact_navit_cpu_preprocessing(image_processor):
        features, actual_grids = materialize_exact_navit_image_features(
            medias, image_processor
        )
    else:
        features, actual_grids = _materialize_kimi_k25_cpu_items_reference(
            items, image_processor
        )

    expected_grids = torch.cat(
        [item.model_specific_data["grid_thws"] for item in items], dim=0
    )
    if not torch.equal(actual_grids.cpu(), expected_grids.cpu()):
        raise ValueError("Kimi-K2.5 deferred CPU preprocessing produced wrong grids")
    return features


def prepare_kimi_k25_encoder_inputs(images, image_processor):
    """Defer exact K2.5 image work until encoder-DP ownership is known."""
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.multimodal.encoder_preprocessing import (
        EncoderPreprocessOutput,
        hash_raw_encoder_item,
    )
    from sglang.srt.multimodal.processors.kimi_k25 import (
        _grid_thw_from_resize_config,
        navit_resize_config,
    )

    config = getattr(image_processor, "media_proc_cfg", None)
    required = (
        "patch_size",
        "merge_kernel_size",
        "in_patch_limit",
        "patch_limit_on_one_side",
    )
    if not isinstance(config, dict) or any(name not in config for name in required):
        raise ValueError("Kimi-K2.5 encoder preprocessing needs NaViT media config")

    patch_size = int(config["patch_size"])
    merge_kernel_size = int(config["merge_kernel_size"])
    items = []
    grids = []
    original_image_sizes = []
    for value in images:
        if isinstance(value, dict):
            if value.get("type") != "image" or "image" not in value:
                raise ValueError(f"Unsupported Kimi-K2.5 encoder media item: {value}")
            value = value["image"]
        image = _to_pil_rgb(value)
        width, height = image.size
        resize_config = navit_resize_config(
            width,
            height,
            patch_size,
            merge_kernel_size,
            int(config["in_patch_limit"]),
            int(config["patch_limit_on_one_side"]),
            config.get("fixed_output_tokens"),
        )
        grid = _grid_thw_from_resize_config(resize_config, patch_size)
        grid_tensor = torch.tensor([grid], dtype=torch.int64)
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=image,
            model_specific_data={"grid_thws": grid_tensor},
        )
        item.set_hash(hash_raw_encoder_item(np.asarray(image)))
        items.append(item)
        grids.append(grid)
        original_image_sizes.append([width, height])

    return EncoderPreprocessOutput(
        {
            "pixel_values": [item.feature for item in items],
            "grid_thws": torch.tensor(grids, dtype=torch.int64),
            "original_image_sizes": original_image_sizes,
        },
        mm_items=items,
        item_sizes=[math.prod(grid) for grid in grids],
        materialize_local_items=functools.partial(
            materialize_kimi_k25_cpu_item_features,
            image_processor=image_processor,
        ),
    )
