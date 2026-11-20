"""Validate Gemma 4 vision input metadata before entering model collectives."""

import torch

from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.utils import flatten_nested_list


def _feature_shape(feature) -> tuple[int, ...]:
    # IPC features need not be reconstructed merely to inspect their shape.
    if isinstance(feature, CudaIpcTensorTransportProxy):
        shape = getattr(feature, "shape", None)  # CUDA VMM transport
        if shape is None:
            shape = feature.proxy_state["ipc_extra"]["recons_shape"]
    elif isinstance(feature, torch.Tensor):
        shape = feature.shape
    else:
        shape = None
    if shape is None:
        raise ValueError("Gemma 4 vision inputs must be tensor features")
    return tuple(shape)


def validate_gemma4_vision_inputs(mm_inputs, config) -> None:
    """Reject malformed pixels/positions using shapes only, never tensor values.

    Full embeddings and featureless PP/cache metadata keep their existing paths.
    The receiver shares any rejection across TP before admitting the request.
    """
    vision_config = config.vision_config
    if vision_config is None:
        return
    patch_width = 3 * vision_config.patch_size**2
    hidden_size = config.text_config.hidden_size
    for item in getattr(mm_inputs, "mm_items", ()):
        if not (item.is_image() or item.is_video()) or item.feature is None:
            continue
        position_key = "image_position_ids" if item.is_image() else "video_position_ids"
        positions = flatten_nested_list([getattr(item, position_key, None)])
        for index, pixels in enumerate(flatten_nested_list([item.feature])):
            shape = _feature_shape(pixels)
            if len(shape) in (2, 3) and shape[-1] == hidden_size:
                continue  # Same precomputed-feature fast path as the model.
            pixel_dims = (3, 4) if item.is_video() else (2, 3)
            if (
                len(shape) not in pixel_dims
                or min(shape) <= 0
                or shape[-1] != patch_width
            ):
                raise ValueError(
                    f"Gemma 4 {item.modality.name.lower()} pixels have shape {shape}; "
                    f"expected [batch, patches, {patch_width}] RGB patches"
                )
            if index >= len(positions) or positions[index] is None:
                raise ValueError(f"Gemma 4 pixels[{index}] is missing {position_key}")
            pos_shape = _feature_shape(positions[index])
            # The video encoder collapses (videos, frames, patches, pixels).
            if item.is_video() and len(shape) == 4:
                shape = (shape[0] * shape[1], *shape[2:])
            position_dims = (3, 4) if item.is_video() else (2, 3)
            if item.is_video() and len(pos_shape) == 4:
                pos_shape = (pos_shape[0] * pos_shape[1], *pos_shape[2:])
            batch_size = shape[0] if len(shape) == 3 else 1
            if (
                len(pos_shape) not in position_dims
                or pos_shape[-1] != 2
                or pos_shape[-2] != shape[-2]
                or (len(pos_shape) == 3 and pos_shape[0] not in (1, batch_size))
            ):
                raise ValueError(
                    f"Gemma 4 {position_key} shape {pos_shape} does not match pixels {shape}"
                )
