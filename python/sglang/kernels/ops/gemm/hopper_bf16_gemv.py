from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Hopper single-token bf16 GEMV (see csrc/gemm/hopper_bf16_gemv.cuh).
# Decode at bs=1 is a pure weight-streaming workload; cuBLAS leaves 5-15%
# DRAM bandwidth on the table for the mid-sized per-layer weights of dense
# hybrid models (measured on H200: 3.2-3.9 TB/s vs a 4.3 TB/s copy ceiling).
# One warp computes a few consecutive rows, the activation vector lives in
# shared memory, and weights are streamed with evict-first loads.

_MAX_K = 17408  # static smem limit (K * 2 bytes <= 48KB) with margin
_MAX_N = 65536  # very large N (lm_head) is already at the bandwidth ceiling


def _config(n: int) -> tuple[int, int, int]:
    """(rows_per_warp, k_unroll, num_warps) tuned on H200."""
    if n >= 8192:
        return (2, 2, 8)
    return (1, 2, 8)


@cache_once
def _jit_hopper_bf16_gemv_module(n: int, k: int) -> Module:
    rows, unroll, warps = _config(n)
    args = make_cpp_args(n, k, rows, unroll, warps)
    return load_jit(
        "hopper_bf16_gemv",
        *args,
        cuda_files=["gemm/hopper_bf16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::HopperBf16GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_hopper_bf16_gemv(m: int, n: int, k: int) -> bool:
    if not (
        m == 1
        and k % 512 == 0
        and 512 <= k <= _MAX_K
        and n % 8 == 0
        and 64 <= n <= _MAX_N
    ):
        return False
    # cuBLAS already runs at ~3.9 TB/s for mid-large N around 16K; the wins
    # concentrate where its tiling underutilizes DRAM (small/odd N) and very
    # wide N. Measured on H200 against cuBLAS 12.x.
    return n < 12288 or n >= 32768


def hopper_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[1, N] = x[1, K] @ w[N, K]^T, all bf16, fp32 accumulation."""
    out = torch.empty((1, w.shape[0]), dtype=x.dtype, device=x.device)
    module = _jit_hopper_bf16_gemv_module(w.shape[0], w.shape[1])
    module.run(x, w, out)
    return out
