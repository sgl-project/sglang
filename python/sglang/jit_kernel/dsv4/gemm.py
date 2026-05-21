import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import cpu_has_amx_support, get_bool_env_var, is_cpu, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu = is_cpu()
_cpu_amx = cpu_has_amx_support()

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if _is_cpu and _cpu_amx:
        return torch.ops.sgl_kernel.weight_packed_linear(
            x, y, None, True, torch.float32
        )
    elif _linear_bf16_fp32_algo == "deep_gemm":
        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    elif _use_aiter:
        return tgemm.mm(x, y, otype=torch.float32)
    else:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
