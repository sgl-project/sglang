"""Bound Qwen image preprocessing temporaries for CPU feature transport."""

import torch
from transformers.image_processing_utils import BatchFeature


def install_cpu_transport_image_chunks(image_processor, chunk_size: int) -> None:
    """Keep GPU transforms, transferring each completed chunk before concatenation.

    The original processor expands each image independently. Its final GPU cat
    otherwise duplicates the entire request's pixel features before SGLang moves
    them to CPU. Only the Qwen2-VL implementation with these two output fields is
    supported; callers must already have selected CPU feature transport.
    """
    if chunk_size <= 0:
        raise ValueError("Qwen image preprocessing chunk size must be positive")
    previous = getattr(image_processor, "_sglang_cpu_transport_image_chunk_size", None)
    if previous is not None:
        if previous != chunk_size:
            raise ValueError("Cannot replace an installed image chunk size")
        return
    original = image_processor._preprocess
    if (
        original.__func__.__module__
        != "transformers.models.qwen2_vl.image_processing_qwen2_vl"
    ):
        raise ValueError(
            "CPU transport image chunks require the Qwen2-VL image processor"
        )

    def bounded_preprocess(images, *args, **kwargs):
        if (
            not isinstance(images, list)
            or len(images) <= chunk_size
            or not all(
                isinstance(image, torch.Tensor) and image.is_cuda for image in images
            )
        ):
            return original(images, *args, **kwargs)
        pixels = []
        grids = []
        for start in range(0, len(images), chunk_size):
            result = original(images[start : start + chunk_size], *args, **kwargs)
            if set(result.keys()) != {"pixel_values", "image_grid_thw"}:
                raise ValueError("Unexpected Qwen image processor output fields")
            pixels.append(result["pixel_values"].cpu())
            grids.append(result["image_grid_thw"].cpu())
            del result
        return BatchFeature(
            data={
                "pixel_values": torch.cat(pixels, dim=0),
                "image_grid_thw": torch.cat(grids, dim=0),
            },
            tensor_type=kwargs.get("return_tensors"),
        )

    image_processor._preprocess = bounded_preprocess
    image_processor._sglang_cpu_transport_image_chunk_size = chunk_size
