"""Validate Gemma 4 vision input metadata before entering model collectives."""

from typing import Union

import torch

from sglang.srt.managers.req_types import MultimodalInputs, MultimodalProcessorOutput
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.multimodal.transport.cuda_vmm import CudaVmmTensorTransportProxy
from sglang.srt.utils import flatten_nested_list


def _feature_shape(feature) -> tuple[int, ...]:
    # Transport proxies carry their shape, so validation never reconstructs them.
    if isinstance(feature, CudaVmmTensorTransportProxy):
        return tuple(feature.shape)
    if isinstance(feature, CudaIpcTensorTransportProxy):
        return tuple(feature.proxy_state["ipc_extra"]["recons_shape"])
    if isinstance(feature, torch.Tensor):
        return tuple(feature.shape)
    raise ValueError("Gemma 4 vision inputs must be tensor features")


def validate_gemma4_vision_inputs(
    mm_inputs: Union[MultimodalInputs, MultimodalProcessorOutput],
    *,
    patch_width: int,
    hidden_size: int,
    image_dims: tuple[int, ...],
    video_dims: tuple[int, ...],
) -> None:
    """Reject malformed pixels/positions using shapes only, never tensor values.

    `image_dims`/`video_dims` are the pixel and position ranks the model's
    encoder accepts. Full embeddings and featureless PP/cache metadata pass.
    """
    for item in mm_inputs.mm_items:
        if not (item.is_image() or item.is_video()) or item.feature is None:
            continue
        position_key = "image_position_ids" if item.is_image() else "video_position_ids"
        dims = image_dims if item.is_image() else video_dims
        positions = flatten_nested_list([item.model_specific_data.get(position_key)])
        for index, pixels in enumerate(flatten_nested_list([item.feature])):
            shape = _feature_shape(pixels)
            if len(shape) in (2, 3) and shape[-1] == hidden_size:
                continue  # Same precomputed-feature fast path as the model.
            if len(shape) not in dims or min(shape) <= 0 or shape[-1] != patch_width:
                raise ValueError(
                    f"Gemma 4 {item.modality.name.lower()} pixels have shape {shape}; "
                    f"expected [batch, patches, {patch_width}] RGB patches"
                )
            if index >= len(positions) or positions[index] is None:
                raise ValueError(f"Gemma 4 pixels[{index}] is missing {position_key}")
            pos_shape = _feature_shape(positions[index])
            if len(pos_shape) not in dims:
                raise ValueError(
                    f"Gemma 4 {position_key} shape {pos_shape} has the wrong rank"
                )
            # The encoders collapse (videos, frames, patches, pixels).
            if len(shape) == 4:
                shape = (shape[0] * shape[1], *shape[2:])
            if len(pos_shape) == 4:
                pos_shape = (pos_shape[0] * pos_shape[1], *pos_shape[2:])
            batch_size = shape[0] if len(shape) == 3 else 1
            if (
                pos_shape[-1] != 2
                or pos_shape[-2] != shape[-2]
                or (len(pos_shape) == 3 and pos_shape[0] not in (1, batch_size))
            ):
                raise ValueError(
                    f"Gemma 4 {position_key} shape {pos_shape} does not match pixels {shape}"
                )
