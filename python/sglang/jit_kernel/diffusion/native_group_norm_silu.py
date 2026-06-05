from __future__ import annotations

import functools
import os
from typing import Any, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

_H200_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_B200_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_SUPPORTED_NUM_GROUPS = {32}
_LARGE_THRESH = int(os.environ.get("GNS_SMALL_LARGE_THRESH", str(1 << 16)))
_CHUNK_ELEMS = 8192
_GIANT_THRESH = int(os.environ.get("GNS_GIANT_THRESH", str(700_000)))
_GIANT_CHUNK_ELEMS = int(os.environ.get("GNS_GIANT_CHUNK", str(16384)))
_CLEAN_CHUNK_ELEMS = int(os.environ.get("GNS_CLEAN_CHUNK", "0"))
_GIANT_STATS_CHUNK_ELEMS = int(os.environ.get("GNS_GIANT_STATS_CHUNK", "0"))


def _capability(t: torch.Tensor) -> tuple[int, int]:
    return torch.cuda.get_device_capability(t.device)


def _is_h200(t: torch.Tensor) -> bool:
    return _capability(t) == (9, 0)


def _is_blackwell(t: torch.Tensor) -> bool:
    return _capability(t)[0] >= 10


def _aligned(t: torch.Tensor) -> bool:
    return t.storage_offset() == 0


def _spatial(x: torch.Tensor) -> int:
    spatial = 1
    for dim in x.shape[2:]:
        spatial *= int(dim)
    return spatial


def _torch_extension_flags() -> tuple[list[str], list[str]]:
    from torch.utils import cpp_extension as tce

    try:
        include_paths = list(tce.include_paths(device_type="cuda"))
        library_paths = list(tce.library_paths(device_type="cuda"))
    except TypeError:
        include_paths = list(tce.include_paths())
        library_paths = list(tce.library_paths())
    ldflags = [f"-L{p}" for p in library_paths]
    ldflags += [f"-Wl,-rpath,{p}" for p in library_paths]
    ldflags += ["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda"]
    return include_paths, ldflags


@cache_once
def _h200_module():
    include_paths, ldflags = _torch_extension_flags()
    return load_jit(
        "diffusion_native_group_norm_silu_h200",
        "v1",
        cuda_files=["diffusion/group_norm_silu_h200.cu"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        header_only=False,
    )


@cache_once
def _b200_module():
    include_paths, ldflags = _torch_extension_flags()
    return load_jit(
        "diffusion_native_group_norm_silu_b200",
        "v1",
        cuda_files=["diffusion/group_norm_silu_b200.cu"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        header_only=False,
    )


def _base_inputs_ok(x: Any, weight: Any, bias: Any, num_groups: Any) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dim() in (2, 3, 4, 5)
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
    )


def _can_use_h200(x: Any, weight: Any, bias: Any, num_groups: Any) -> bool:
    return (
        _base_inputs_ok(x, weight, bias, num_groups)
        and _is_h200(x)
        and x.dtype in _H200_SUPPORTED_DTYPES
        and x.is_contiguous()
        and _aligned(x)
    )


def _can_use_b200(x: Any, weight: Any, bias: Any, num_groups: Any) -> bool:
    return (
        _base_inputs_ok(x, weight, bias, num_groups)
        and _is_blackwell(x)
        and x.dtype in _B200_SUPPORTED_DTYPES
    )


def can_use_native_group_norm_silu(
    x: Any,
    weight: Any,
    bias: Any,
    num_groups: Any,
) -> bool:
    return _can_use_h200(x, weight, bias, num_groups) or _can_use_b200(
        x, weight, bias, num_groups
    )


def _group_norm_silu_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    return x.new_empty(x.shape)


@functools.lru_cache(maxsize=256)
def _giant_chunk_for(spatial: int) -> int:
    target = _GIANT_CHUNK_ELEMS
    k_min = -(-spatial // target)
    for k in range(k_min, min(4 * k_min, spatial) + 1):
        if spatial % k == 0 and (spatial // k) % 8 == 0:
            return spatial // k
    return target


_row_counters: dict = {}


def _row_counter(num_rows: int, device: torch.device) -> torch.Tensor:
    stream = torch.cuda.current_stream(device)
    key = (device.type, device.index, stream.cuda_stream)
    buf = _row_counters.get(key)
    if buf is None or buf.numel() < num_rows:
        buf = torch.zeros(max(num_rows, 64), dtype=torch.int32, device=device)
        _row_counters[key] = buf
    return buf


def _run_h200_group_norm_silu(
    x3: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    y3: torch.Tensor,
    num_groups: int,
    eps: float,
) -> None:
    mod = _h200_module()
    channels = x3.shape[1]
    spatial = x3.shape[2]
    group_size = (channels // num_groups) * spatial
    with torch.cuda.device(x3.device):
        if group_size < _LARGE_THRESH:
            mod.gns_candidate_small(x3, weight, bias, int(num_groups), float(eps), y3)
            return

        num_rows = x3.shape[0] * num_groups
        clean_chunk = _CLEAN_CHUNK_ELEMS or _giant_chunk_for(spatial)
        use_clean = (
            group_size >= _GIANT_THRESH
            and spatial >= clean_chunk
            and spatial % clean_chunk == 0
        )
        use_giant = (
            not use_clean
            and group_size >= _GIANT_THRESH
            and spatial >= _GIANT_CHUNK_ELEMS
        )
        if use_clean:
            total = num_rows * (group_size // clean_chunk)
        elif use_giant:
            apply_chunk = _giant_chunk_for(spatial)
            stats_chunk = _GIANT_STATS_CHUNK_ELEMS or apply_chunk
            stats_chunks_per_row = (group_size + stats_chunk - 1) // stats_chunk
            total = num_rows * stats_chunks_per_row
        else:
            chunks_per_row = (group_size + _CHUNK_ELEMS - 1) // _CHUNK_ELEMS
            total = num_rows * chunks_per_row

        scratch_kwargs = {"device": x3.device, "dtype": torch.float32}
        partial_sum = torch.empty(total, **scratch_kwargs)
        partial_sumsq = torch.empty(total, **scratch_kwargs)
        mean = torch.empty(num_rows, **scratch_kwargs)
        rstd = torch.empty(num_rows, **scratch_kwargs)
        if use_clean:
            mod.gns_candidate_clean_giant(
                x3,
                weight,
                bias,
                partial_sum,
                partial_sumsq,
                mean,
                rstd,
                _row_counter(num_rows, x3.device),
                int(num_groups),
                float(eps),
                int(clean_chunk),
                y3,
            )
        elif use_giant:
            mod.gns_candidate_giant(
                x3,
                weight,
                bias,
                partial_sum,
                partial_sumsq,
                mean,
                rstd,
                _row_counter(num_rows, x3.device),
                int(num_groups),
                float(eps),
                int(stats_chunk),
                int(apply_chunk),
                y3,
            )
        else:
            mod.gns_candidate_large(
                x3,
                weight,
                bias,
                partial_sum,
                partial_sumsq,
                mean,
                rstd,
                int(num_groups),
                float(eps),
                y3,
            )


@register_custom_op(
    op_name="diffusion_native_group_norm_silu_cuda",
    fake_impl=_group_norm_silu_fake,
)
def _native_group_norm_silu_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    if _is_blackwell(x):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        _b200_module().group_norm_silu(
            x, weight, bias, int(num_groups), float(eps), out
        )
        return out

    batch, channels = int(x.shape[0]), int(x.shape[1])
    spatial = _spatial(x)
    x3 = x.reshape(batch, channels, spatial)
    y3 = torch.empty_like(x3)
    _run_h200_group_norm_silu(x3, weight, bias, y3, int(num_groups), float(eps))
    return y3.reshape_as(x)


def try_native_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> Optional[torch.Tensor]:
    if can_use_native_group_norm_silu(x, weight, bias, num_groups):
        try:
            return _native_group_norm_silu_cuda(x, weight, bias, num_groups, eps)
        except Exception:
            return None
    return None


__all__ = [
    "can_use_native_group_norm_silu",
    "try_native_group_norm_silu",
]
