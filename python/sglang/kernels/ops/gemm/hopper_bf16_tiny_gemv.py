from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Tiny-N bf16 GEMM for decode (see csrc/gemm/hopper_bf16_tiny_gemv.cuh): the
# per-rank KDA beta / gated-attention projections are [8, 4096] at TP8, which
# cuBLAS runs as a split-K GEMM plus a reduce kernel (~6 us for 64 KB). One
# block per output row finishes in the launch latency.

_MAX_M = 4  # x is re-read from global per token; past 4 tokens cuBLAS wins
_MAX_N = 64


def _max_m_bucket(m: int) -> int:
    return 4 if m <= 4 else 8 if m <= 8 else 16


@cache_once
def _jit_hopper_bf16_tiny_gemv_module(max_m: int) -> Module:
    args = make_cpp_args(max_m)
    return load_jit(
        "hopper_bf16_tiny_gemv",
        *args,
        cuda_files=["gemm/hopper_bf16_tiny_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::HopperBf16TinyGemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_hopper_bf16_tiny_gemv(m: int, n: int, k: int) -> bool:
    return 1 <= m <= _MAX_M and 1 <= n <= _MAX_N and k % 1024 == 0 and k <= 16384


def hopper_bf16_tiny_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[M, N] = x[M, K] @ w[N, K]^T, all bf16, fp32 accumulation."""
    m = x.shape[0]
    out = torch.empty((m, w.shape[0]), dtype=torch.bfloat16, device=x.device)
    module = _jit_hopper_bf16_tiny_gemv_module(_max_m_bucket(m))
    module.run(x, w, out)
    return out
