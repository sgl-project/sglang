from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
_JIT_GEMM_WEIGHT_CACHE_ATTR = "_sglang_bf16xfp32_weight_cache"


def _can_use_jit_gemm_bf16xfp32(
    x: torch.Tensor, y: torch.Tensor, *, min_m: int = 8
) -> bool:
    if x.dim() != 2 or y.dim() != 2 or x.shape[1] != y.shape[1]:
        return False
    if x.shape[0] < min_m:
        return False
    if not (x.is_cuda and y.is_cuda):
        return False
    if x.dtype != torch.bfloat16 or y.dtype != torch.float32:
        return False
    if not (x.is_contiguous() and y.is_contiguous()):
        return False
    if y.shape[0] % 64 != 0:
        return False

    from sglang.jit_kernel.gemm_bf16xfp32 import is_gemm_bf16xfp32_supported

    return is_gemm_bf16xfp32_supported(x.device)


def _get_bf16xfp32_weight_split(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.jit_kernel.gemm_bf16xfp32 import split_fp32_weight

    cache_key = (
        y.data_ptr(),
        y._version,
        tuple(y.shape),
        tuple(y.stride()),
        y.device.index,
        y.dtype,
    )
    cache = getattr(y, _JIT_GEMM_WEIGHT_CACHE_ATTR, None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2]

    with torch.no_grad():
        w_high, w_low = split_fp32_weight(y)
    setattr(y, _JIT_GEMM_WEIGHT_CACHE_ATTR, (cache_key, w_high, w_low))
    return w_high, w_low


def _linear_bf16_fp32_cublas(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    return torch.mm(x.float(), y.float().t())


def _linear_bf16_fp32_jit(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    min_m: int = 8,
) -> Optional[torch.Tensor]:
    if not _can_use_jit_gemm_bf16xfp32(x, y, min_m=min_m):
        return None

    from sglang.jit_kernel.gemm_bf16xfp32 import gemm_bf16xfp32

    w_high, w_low = _get_bf16xfp32_weight_split(y)
    return gemm_bf16xfp32(x, w_high, w_low)


def linear_bf16_fp32(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    jit_kernel_min_m: Optional[int] = None,
) -> torch.Tensor:
    if _use_aiter and y.dtype == torch.bfloat16:
        return tgemm.mm(x, y, otype=x.dtype).float()
    elif jit_kernel_min_m is not None:
        output = _linear_bf16_fp32_jit(x, y, min_m=jit_kernel_min_m)
        if output is not None:
            return output
        return _linear_bf16_fp32_cublas(x, y)
    elif _linear_bf16_fp32_algo == "jit":
        output = _linear_bf16_fp32_jit(x, y)
        if output is not None:
            return output
        return _linear_bf16_fp32_cublas(x, y)
    elif _linear_bf16_fp32_algo == "deep_gemm" and y.dtype == torch.bfloat16:
        from sglang.srt.layers import deep_gemm_wrapper

        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    else:
        return _linear_bf16_fp32_cublas(x, y)
