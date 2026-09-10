# SPDX-License-Identifier: Apache-2.0
"""Explicit static-FP8 input; row scales never replace the layer's scalar."""

from typing import Optional, Tuple, Union

import msgspec
import torch


class ModelOptFp8Input(msgspec.Struct, frozen=True):
    """Prequantized activations using the layer's static input scale.

    scale must be layer.input_scale; optional row_scales repeats that scalar
    as FP32 [M, 1], where M is the flattened row count. orig_dtype is the
    linear output dtype. The producer is responsible for scale consistency.
    """

    qx: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    row_scales: Optional[torch.Tensor] = None


ModelOptFp8LinearInput = Union[
    torch.Tensor,
    ModelOptFp8Input,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.dtype],
]


def normalize_and_validate_modelopt_fp8_input(value: object, layer) -> ModelOptFp8Input:
    # Tuple inputs omit row scales; infer a missing output dtype from the layer.
    if isinstance(value, tuple):
        if len(value) not in (2, 3):
            raise ValueError("Expected (qx, scalar[, original dtype])")
        value = ModelOptFp8Input(
            value[0], value[1], value[2] if len(value) == 3 else layer.orig_dtype
        )
    if not isinstance(value, ModelOptFp8Input):
        raise TypeError("Expected a ModelOptFp8Input or a two-/three-field tuple")
    qx, scale, dtype, rows = value.qx, value.scale, value.orig_dtype, value.row_scales
    if not isinstance(qx, torch.Tensor) or qx.dtype != torch.float8_e4m3fn:
        raise TypeError("Static FP8 input must contain an E4M3FN tensor")
    if not qx.is_cuda or qx.ndim < 2 or not qx.is_contiguous():
        raise ValueError("Static FP8 input must be a contiguous CUDA matrix or batch")
    if qx.shape[-1] == 0:
        raise ValueError("Static FP8 input must have a nonzero feature dimension")
    if scale is not layer.input_scale:
        raise ValueError("Static FP8 input must retain the layer.input_scale object")
    if scale.numel() != 1 or scale.dtype != torch.float32 or scale.device != qx.device:
        raise ValueError("Expected the original FP32 scalar on the input device")
    if dtype not in (torch.float16, torch.bfloat16) or dtype != layer.orig_dtype:
        raise ValueError("Static FP8 input must retain the layer's original dtype")
    if qx.shape[-1] != layer.weight.shape[0] or qx.device != layer.weight.device:
        raise ValueError("Static FP8 input is incompatible with the linear weight")
    if rows is not None:
        if not isinstance(rows, torch.Tensor) or (
            rows.shape != (qx.numel() // qx.shape[-1], 1)
            or rows.dtype != torch.float32
            or rows.device != qx.device
            or not rows.is_contiguous()
        ):
            raise ValueError("Expected contiguous FP32 [M, 1] row scales")
    # Row-scale values must match the scalar; avoid a hot-path GPU readback.
    return value
