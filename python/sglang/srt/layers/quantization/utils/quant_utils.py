# Adapted from https://github.com/vllm-project/vllm/model_executor/layers/quantization/utils/quant_utils.py
# SPDX-License-Identifier: Apache-2.0
"""This file is used for /tests and /benchmarks"""

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
from torch import fx


# Use proxy as NamedTuple direct subclasses cannot have static members
class _GroupShape(NamedTuple):
    row: int
    col: int


class GroupShape(_GroupShape):
    """
    This class describes the quantization group shape.
    It includes static members for common shapes (per-tensor, per-token).
    """

    # Aliases for common quantization group shapes
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)


@dataclass(frozen=True)
class ScaleDesc:
    """
    Class for describing a single quantization scaling factor.
    dtype: data type of the scale
    static: static scale if True, dynamic if False
    group_shape: group shape of the scale
    """

    dtype: torch.dtype
    static: bool
    group_shape: GroupShape

    def __str__(self):
        group_shape = (
            "per_tensor"
            if self.group_shape == GroupShape.PER_TENSOR
            else (
                "per_token"
                if self.group_shape == GroupShape.PER_TOKEN
                else str(self.group_shape)
            )
        )

        return (
            f"{fx.graph.dtype_abbrs[self.dtype]},"
            f"{'static' if self.static else 'dynamic'},{group_shape}"
        )


@dataclass(frozen=True)
class QuantKey:
    """
    Class for identifying the type of quantization.
    dtype: quantized data type
    scale: scale descriptor
    scale2: second-level scale descriptor
    symmetric: symmetric if True, asymmetric if False
    """

    dtype: torch.dtype
    scale: ScaleDesc
    scale2: ScaleDesc | None = None
    symmetric: bool = True

    def __str__(self):
        scale2_str = f"scale2({self.scale2})," if self.scale2 else ""
        return (
            f"QuantKey({fx.graph.dtype_abbrs[self.dtype]},"
            f"scale({self.scale}),{scale2_str}"
            f"{'a' if not self.symmetric else ''}symmetric)"
        )


# Normalize the group_shape to the full extent for any dims that are -1
def _normalize_quant_group_shape(x: torch.Tensor, group_shape: GroupShape):
    # -1 means full extent
    return (
        group_shape[0] if group_shape[0] > 0 else x.shape[-2],
        group_shape[1] if group_shape[1] > 0 else x.shape[-1],
    )


# Useful when treating N-dimensional group scaling as extended numpy-style
# broadcasting in numpy simply stretches dimensions with an extent of 1 to match
# the target shape by repeating the data along that dimension (broadcasting)
# , we extend these semantics to say if the extent of a dimension in the
# source shape is not 1 and does not match the target shape we repeat each
# element along that dimension src_shape[dim] // target_shape[dim] times
# example if we have:
#       a = [[1, 2], and target_shape = (2, 4)
#            [3, 4]]
# then we would expand a to:
#       a = [[1, 1, 2, 2],
#            [3, 3, 4, 4]]
# NOTE this function does not explicitly broadcast dimensions
# with an extent of 1, since this can be done implicitly by pytorch
def group_broadcast(t, shape):
    for i, s in enumerate(shape):
        if t.shape[i] != s and t.shape[i] != 1:
            assert s % t.shape[i] == 0
            t = (
                t.unsqueeze(i + 1)
                .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                .flatten(i, i + 1)
            )
    return t


# Quantize assuming once scale per group of elements with shape group_shape,
# example group shapes:
#  * (-1, -1)   for per-tensor quantization
#  * (1, -1)    for per-row quantization
#  * (-1, 1)    for per-column quantization
#  * (128, 128) for 128x128 deepseek style block quantization
#  * (1, 128)   for deepseek style activation quantization
#               (i.e. per-token-per-group)
def scaled_quantize(
    x: torch.Tensor,
    group_shape: GroupShape,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_shape = _normalize_quant_group_shape(x, group_shape)
    assert quant_dtype.is_floating_point, (
        "currently `scaled_quantize` only supports floating point dtypes "
        "but could be extended to support other dtypes"
    )

    finfo = torch.finfo(quant_dtype)

    # Reshape (M, N) into (BLK_M, BLOCK_SIZE_M, BLK_N, BLOCK_SIZE_N)
    assert x.ndim == 2
    assert x.shape[0] % group_shape[0] == 0 and x.shape[1] % group_shape[1] == 0
    blk_m, blk_n = x.shape[0] // group_shape[0], x.shape[1] // group_shape[1]
    x_blkd = x.reshape(blk_m, group_shape[0], blk_n, group_shape[1])

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    x_blkd_permd = x_blkd.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    x_blkd_permd = x_blkd_permd.flatten(start_dim=2)

    # Compute scales
    min_val, max_val = x_blkd_permd.aminmax(dim=-1)
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax

    # Apply scale and convert form:
    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    x_scl_sat = (
        (x_blkd_permd * scale.unsqueeze(-1))
        .clamp(min=finfo.min, max=finfo.max)
        .reshape(blk_m, blk_n, group_shape[0], group_shape[1])
        .permute(0, 2, 1, 3)
        .reshape(x.shape)
    )

    return x_scl_sat.to(quant_dtype).contiguous(), scale.float().reciprocal()


# inverses `scaled_quantize`
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: GroupShape | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_shape is not None:
        group_shape = _normalize_quant_group_shape(x_q, group_shape)

    if x_s.ndim == 0:  # scalar
        x_s = x_s.unsqueeze(-1).unsqueeze(-1)  # convert to (1, 1) tensor
    if x_s.ndim == 1:
        if group_shape is None:
            raise AssertionError(
                "if x_s is 1D tensor, group_shape must be provided otherwise "
                "its ambiguous which dimension to broadcast x_s to"
            )
        # unsqueeze the scales for the dimension where we want to broadcast
        # across the full extent
        if group_shape[0] == x_q.shape[-2]:
            x_s = x_s.unsqueeze(-2)
        elif group_shape[1] == x_q.shape[-1]:
            x_s = x_s.unsqueeze(-1)
        else:
            raise AssertionError(
                "if x_s is a vector we should be broadcasting it to the full "
                "extent of one of the dimensions"
            )

    if group_shape is not None:
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
    x_s = group_broadcast(x_s.to(torch.float32), x_q.shape)
    return (x_q.to(torch.float32) * x_s).to(out_dtype)
