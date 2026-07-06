import functools
import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
_HPC_OPS_BF16XFP32_SCALE = 1.0 / 256.0
_HPC_OPS_WEIGHT_CACHE_ATTR = "_sglang_hpc_ops_bf16xfp32_cache"


@functools.lru_cache(maxsize=1)
def _get_hpc_ops_module():
    try:
        import hpc

        return hpc
    except Exception as exc:
        logger.warning(
            "SGLANG_OPT_BF16_FP32_GEMM_ALGO=hpc_ops requested, but importing "
            "HPC-Ops failed (%s). Falling back to the cublas path.",
            exc,
        )
        return None


def _can_use_hpc_ops_bf16xfp32(
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
    if torch.cuda.get_device_capability(x.device)[0] != 9:
        return False
    return _get_hpc_ops_module() is not None


def _get_hpc_ops_bf16xfp32_weight_split(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        y.data_ptr(),
        y._version,
        tuple(y.shape),
        tuple(y.stride()),
        y.device.index,
        y.dtype,
    )
    cache = getattr(y, _HPC_OPS_WEIGHT_CACHE_ATTR, None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2]

    with torch.no_grad():
        w_high = y.to(torch.bfloat16).contiguous()
        w_low = ((y - w_high.float()) / _HPC_OPS_BF16XFP32_SCALE).to(torch.bfloat16)
        w_low = w_low.contiguous()
    setattr(y, _HPC_OPS_WEIGHT_CACHE_ATTR, (cache_key, w_high, w_low))
    return w_high, w_low


def _linear_bf16_fp32_cublas(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    return torch.mm(x.float(), y.float().t())


def _linear_bf16_fp32_hpc_ops(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    use_splitk: bool = True,
    split_flag: Optional[torch.Tensor] = None,
    min_m: int = 8,
) -> Optional[torch.Tensor]:
    if not _can_use_hpc_ops_bf16xfp32(x, y, min_m=min_m):
        return None
    hpc = _get_hpc_ops_module()
    w_high, w_low = _get_hpc_ops_bf16xfp32_weight_split(y)
    return hpc.gemm_bf16xfp32(
        x,
        w_high,
        w_low,
        _HPC_OPS_BF16XFP32_SCALE,
        True,
        use_splitk,
        split_flag,
    )


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if _use_aiter and y.dtype == torch.bfloat16:
        return tgemm.mm(x, y, otype=x.dtype).float()
    elif _linear_bf16_fp32_algo == "hpc_ops":
        output = _linear_bf16_fp32_hpc_ops(x, y)
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
