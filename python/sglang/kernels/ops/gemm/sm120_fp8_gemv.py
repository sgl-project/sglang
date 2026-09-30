from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# SM120 single-token per-tensor-scale FP8 GEMV
# (see csrc/gemm/sm120_fp8_gemv.cuh).
#
# On consumer/workstation Blackwell, cuBLAS serves M=1 fp8 GEMMs with SM89
# tiles that reach only 50-70% of DRAM bandwidth for mid-sized N (a ~19us
# wave-quantization floor). A warp-per-row streaming GEMV with evict-first
# weight loads recovers most of the gap for the decode hot path.

_MAX_K = 32768  # static smem limit: K bytes <= 48KB with margin
_MAX_N = 32768  # very large N is served well by cuBLAS already


def _config(n: int) -> tuple[int, int, int]:
    """(rows_per_warp, k_unroll, num_warps)."""
    if n >= 8192:
        return (2, 1, 8)
    return (1, 1, 8)


@cache_once
def _jit_sm120_fp8_gemv_module(n: int, k: int) -> Module:
    rows, unroll, warps = _config(n)
    args = make_cpp_args(n, k, rows, unroll, warps)
    return load_jit(
        "sm120_fp8_gemv",
        *args,
        cuda_files=["gemm/sm120_fp8_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Fp8GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_sm120_fp8_gemv(m: int, n: int, k: int) -> bool:
    return (
        m == 1
        and k % 512 == 0
        and 512 <= k <= _MAX_K
        and n % 16 == 0
        and 256 <= n <= _MAX_N
    )


def sm120_fp8_gemv(
    x_fp8: torch.Tensor, w_fp8: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """y[1, N] = (x[1, K] @ w[N, K]^T) * alpha; fp8 e4m3 in, bf16 out."""
    out = torch.empty((1, w_fp8.shape[0]), dtype=torch.bfloat16, device=x_fp8.device)
    module = _jit_sm120_fp8_gemv_module(w_fp8.shape[0], w_fp8.shape[1])
    module.run(x_fp8, w_fp8, alpha, out)
    return out
