import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
    if algo == "deep_gemm":
        import deep_gemm

        z = x.new_empty(x.size(0), y.size(0), dtype=torch.float32)
        deep_gemm.bf16_gemm_nt(x, y, z)
        return z
    elif _use_aiter:
        return tgemm.mm(x, y, otype=torch.float32)
    else:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
