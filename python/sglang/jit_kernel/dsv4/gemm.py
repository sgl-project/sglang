import functools
import importlib.util
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
_HPC_GEMM_WEIGHT_CACHE_ATTR = "_sglang_bf16xfp32_weight_cache"
# The HPC-Ops bf16xfp32 GEMM consumes the fp32 weight decomposed into two
# bf16 halves: w_high = w.bf16 and w_low = ((w - w_high) / scale).bf16 with
# scale = 1/256, so that w ~= w_high + scale * w_low.
_HPC_GEMM_WEIGHT_SCALE = 1.0 / 256.0
# Set once the first weight split is cached; never reset. Online weight
# updates are rejected while True (see bf16xfp32_weight_split_cache_active).
_split_cache_created = False


@functools.cache
def _hpc_gemm_bf16xfp32_available() -> bool:
    """HPC-Ops (https://github.com/Tencent/hpc-ops) ships sm90a kernels."""
    if importlib.util.find_spec("hpc") is None:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _can_use_hpc_gemm_bf16xfp32(
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
    return _hpc_gemm_bf16xfp32_available()


def _get_bf16xfp32_weight_split(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the fp32 weight for the HPC-Ops kernel and cache the result
    (plus the split-K flag workspace, which the kernel leaves zeroed) on the
    weight tensor.

    The cache key covers the weight's layout identity only, NOT its content
    or ``_version``: weight content changes cannot be tracked reliably (the
    SGLang loaders write in place via ``param.data.copy_()``, which bumps
    nothing), and captured CUDA graphs hold the split buffers' addresses, so
    reallocating on a key change would silently break replay. Instead, the
    split is computed once and online weight updates are rejected while a
    cache is active (see bf16xfp32_weight_split_cache_active()).
    """
    import hpc

    global _split_cache_created

    cache_key = (
        y.data_ptr(),
        tuple(y.shape),
        tuple(y.stride()),
        y.device.index,
        y.dtype,
    )
    cache = getattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2], cache[3]

    with torch.no_grad():
        w_high = y.to(torch.bfloat16)
        w_low = ((y - w_high.float()) / _HPC_GEMM_WEIGHT_SCALE).to(torch.bfloat16)
    split_flag = hpc.get_gemm_bf16xfp32_workspace(y.shape[0])
    setattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, (cache_key, w_high, w_low, split_flag))
    _split_cache_created = True
    return w_high, w_low, split_flag


def bf16xfp32_weight_split_cache_active() -> bool:
    """Whether a bf16xfp32 weight split has been cached in this process.

    The online weight-update APIs must reject updates while this is true:
    the split would keep serving the old weights (in-place loader writes are
    invisible to the cache, and captured CUDA graphs replay the split
    buffers by address). Deliberately sticky (never reset): weights live for
    the server's lifetime, and a stale True only rejects an update that the
    optimization does not support anyway.
    """
    return _split_cache_created


def _linear_bf16_fp32_cublas(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    return torch.mm(x.float(), y.float().t())


def _linear_bf16_fp32_hpc(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    min_m: int = 8,
) -> Optional[torch.Tensor]:
    if not _can_use_hpc_gemm_bf16xfp32(x, y, min_m=min_m):
        return None

    import hpc

    w_high, w_low, split_flag = _get_bf16xfp32_weight_split(y)
    return hpc.gemm_bf16xfp32(
        x,
        w_high,
        w_low,
        _HPC_GEMM_WEIGHT_SCALE,
        use_fp32_output=True,
        use_splitk=True,
        split_flag=split_flag,
    )


def linear_bf16_fp32(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hpc_kernel_min_m: Optional[int] = None,
) -> torch.Tensor:
    if _use_aiter and y.dtype == torch.bfloat16:
        return tgemm.mm(x, y, otype=x.dtype).float()
    elif hpc_kernel_min_m is not None:
        output = _linear_bf16_fp32_hpc(x, y, min_m=hpc_kernel_min_m)
        if output is not None:
            return output
        return _linear_bf16_fp32_cublas(x, y)
    elif _linear_bf16_fp32_algo == "hpc":
        output = _linear_bf16_fp32_hpc(x, y)
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
