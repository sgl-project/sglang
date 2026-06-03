from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_SUPPORTED_NUM_GROUPS = {32}
_USE_PDL = False
_LARGE_THRESH = 1 << 16
_CHUNK_ELEMS = 8192
_GIANT_THRESH = 900_000


def _is_h200(t: torch.Tensor) -> bool:
    return torch.cuda.get_device_capability(t.device) == (9, 0)


def _aligned(t: torch.Tensor) -> bool:
    return t.storage_offset() == 0


def _spatial(x: torch.Tensor) -> int:
    spatial = 1
    for dim in x.shape[2:]:
        spatial *= int(dim)
    return spatial


@cache_once
def _module(dtype: torch.dtype, pdl: bool):
    args = make_cpp_args(dtype, pdl)
    return load_jit(
        "diffusion_native_group_norm_silu",
        *args,
        cuda_files=["diffusion/group_norm_silu_native.cuh"],
        cuda_wrappers=[
            ("group_norm_silu", f"GroupNormSiluKernel<{args}>::run"),
            ("group_norm_silu_large", f"GroupNormSiluKernel<{args}>::run_large"),
        ],
    )


def can_use_native_group_norm_silu(
    x: Any,
    weight: Any,
    bias: Any,
    num_groups: Any,
) -> bool:
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and _is_h200(x)
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dtype in _SUPPORTED_DTYPES
        and x.dim() in (2, 3, 4, 5)
        and x.is_contiguous()
        and _aligned(x)
        and isinstance(num_groups, int)
        and num_groups in _SUPPORTED_NUM_GROUPS
        and x.shape[1] % num_groups == 0
        and isinstance(weight, torch.Tensor)
        and isinstance(bias, torch.Tensor)
        and weight.device == x.device
        and bias.device == x.device
        and weight.dtype == x.dtype
        and bias.dtype == x.dtype
        and weight.dim() == 1
        and bias.dim() == 1
        and tuple(weight.shape) == (x.shape[1],)
        and tuple(bias.shape) == (x.shape[1],)
        and weight.is_contiguous()
        and bias.is_contiguous()
        and _aligned(weight)
        and _aligned(bias)
    ):
        return False
    group_size = (x.shape[1] // int(num_groups)) * _spatial(x)
    return group_size < _GIANT_THRESH


@register_custom_op(op_name="diffusion_native_group_norm_silu_cuda", out_shape="x")
def _native_group_norm_silu_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    batch, channels = int(x.shape[0]), int(x.shape[1])
    spatial = _spatial(x)
    group_size = (channels // int(num_groups)) * spatial
    x3 = x.reshape(batch, channels, spatial)
    y3 = torch.empty_like(x3)
    mod = _module(x.dtype, _USE_PDL)
    if group_size >= _LARGE_THRESH:
        num_rows = batch * int(num_groups)
        chunks_per_row = (group_size + _CHUNK_ELEMS - 1) // _CHUNK_ELEMS
        total_tasks = num_rows * chunks_per_row
        scratch_kwargs = {"device": x.device, "dtype": torch.float32}
        partial_sum = torch.empty(total_tasks, **scratch_kwargs)
        partial_sumsq = torch.empty(total_tasks, **scratch_kwargs)
        mean = torch.empty(num_rows, **scratch_kwargs)
        rstd = torch.empty(num_rows, **scratch_kwargs)
        mod.group_norm_silu_large(
            x3,
            weight,
            bias,
            y3,
            partial_sum,
            partial_sumsq,
            mean,
            rstd,
            int(num_groups),
            float(eps),
        )
    else:
        mod.group_norm_silu(x3, weight, bias, y3, int(num_groups), float(eps))
    return y3.reshape_as(x)


def try_native_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> Optional[torch.Tensor]:
    if can_use_native_group_norm_silu(x, weight, bias, num_groups):
        return _native_group_norm_silu_cuda(x, weight, bias, num_groups, eps)
    return None


__all__ = [
    "can_use_native_group_norm_silu",
    "try_native_group_norm_silu",
]
