import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if _use_aiter:
        # Request fp32 output directly instead of computing a bf16 GEMM and then
        # casting with `.float()`. The explicit cast shows up as a separate
        # `bfloat16tofloat32_copy` kernel per call in the DSV4 indexer
        # (compute_kv_score) path; emitting fp32 from the GEMM epilogue removes
        # that copy and also avoids rounding the result to bf16 first.
        # Requires AITER to expose a native fp32-output tuned kernel for these
        # shapes (companion AITER change); otherwise tgemm falls back internally.
        return tgemm.mm(x, y, otype=torch.float32)
    elif _linear_bf16_fp32_algo == "deep_gemm":
        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    else:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
