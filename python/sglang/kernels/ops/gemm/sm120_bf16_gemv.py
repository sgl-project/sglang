from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# SM120 (RTX 5090 / RTX PRO 6000) dense BF16 GEMV for skinny decode (M <= 4).
#
# cuBLAS serves M in {1,2,4} bf16 GEMMs with SM80 tiles that reach only a
# fraction of DRAM bandwidth on consumer Blackwell (measured 0.19-2.3 TB/s for
# the gemma-4-E2B projections). A warp-per-row streaming GEMV with shared-memory
# activations and evict-first weight loads recovers most of the gap
# (1.5-4.6 TB/s), so the decode hot path wins by 1.3-2x. fp32 accumulation.

_MAX_K = 28672  # M<=2: M*K*2B <= 112KB dynamic smem (with 220KB opt-in)
_MAX_N = 65536  # very wide N (lm_head) is already well served by cuBLAS


def _config(n: int, k: int, m: int) -> tuple[int, int]:
    """(rows_per_warp, num_warps). Tuned on RTX 5090 per (M, N, K).

    M=2 wins on 2 rows/warp for very wide N (halves the block count for the
    gate_up grid) but 1 row/warp elsewhere. M=4 wants 8 warps for wide N /
    large K (more parallel streams) but 4 warps for narrow N (fewer blocks
    contending for L2). Wide-N thresholds sit below gate_up (N=12288);
    large-K thresholds sit below the down-proj K=6144.
    """
    if m == 4:
        if n >= 8192 or k >= 4096:
            return (1, 8)
        return (1, 4)
    if m == 2:
        if n >= 8192:
            return (2, 4)
        return (1, 4)
    # M == 1
    if n >= 4096:
        return (2, 8)
    return (1, 8)


# (n, k, m) shapes whose JIT module is already compiled+loaded in this
# process. load_jit (filesystem + subprocess + dlopen) and the one-time
# cudaFuncSetAttribute inside run() are both illegal inside CUDA graph
# capture, so a capture-time call is only legal for a warmed-up shape.
_LOADED_SHAPES: set[tuple[int, int, int]] = set()


@cache_once
def _jit_sm120_bf16_gemv_module(n: int, k: int, m: int) -> Module:
    rows, warps = _config(n, k, m)
    args = make_cpp_args(n, k, m, rows, warps)
    module = load_jit(
        "sm120_bf16_gemv",
        *args,
        cuda_files=["gemm/sm120_bf16_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Sm120Bf16GemvKernel<{args}>::run"),
            ("warmup", f"sglang::Sm120Bf16GemvKernel<{args}>::warmup"),
        ],
        extra_cuda_cflags=["-O3"],
    )
    # Opt in to >48KB dynamic smem now (a no-op for smaller K): doing it at
    # load time keeps capture-time run() calls free of module mutations.
    module.warmup()
    _LOADED_SHAPES.add((n, k, m))
    return module


def use_sm120_bf16_gemv(m: int, n: int, k: int) -> bool:
    return (
        m in (1, 2, 4)
        and k % 256 == 0
        and 512 <= k <= _MAX_K
        and n % 8 == 0
        and 64 <= n <= _MAX_N
    )


def sm120_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[M, N] = x[M, K] @ w[N, K]^T, all bf16, fp32 accumulation (M <= 4)."""
    x = x.contiguous()
    w = w.contiguous()
    m, k = x.shape
    n = w.shape[0]
    import os as _os
    if _os.environ.get("SGLANG_GEMV_DEBUG") == "1":
        import sys
        print(f"[gemv-call] m={m} n={n} k={k} capturing={torch.cuda.is_current_stream_capturing()} loaded={(n,k,m) in _LOADED_SHAPES}", file=sys.stderr, flush=True)
    if torch.cuda.is_current_stream_capturing() and (n, k, m) not in _LOADED_SHAPES:
        raise RuntimeError(
            f"sm120_bf16_gemv: (M={m}, N={n}, K={k}) has never been run or "
            "warmed up outside CUDA graph capture. The first call must load "
            "the JIT module and cudaFuncSetAttribute the >48KB dynamic smem "
            "opt-in, both of which are illegal during capture. Call "
            "warmup_sm120_bf16_gemv(m, n, k) (or run one eager "
            "sm120_bf16_gemv on this shape) before capture."
        )
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    module = _jit_sm120_bf16_gemv_module(n, k, m)
    module.run(x, w, out)
    return out


# gemma-4-E2B decode projections: (N, K) for every gated M in {1, 2, 4}.
# Servers that capture decode CUDA graphs can pre-warm these at model-init
# time with prewarm_sm120_bf16_gemv_e2b() so capture never hits the lazy
# JIT-load path even when a shape had no prior eager call.
_E2B_NK = (
    (2048, 1536),
    (512, 1536),
    (1536, 2048),
    (12288, 1536),
    (1536, 6144),
    # gemma-4-E2B is a per-layer model (hidden_size_per_layer_input=256); its
    # decode also runs these projections (per-layer proj, fused q/kv, fused
    # gate_up, and the larger per-layer-output down projections):
    (256, 1536),
    (2560, 1536),
    (5120, 1536),
    (8960, 1536),
    (24576, 1536),
    (1536, 4096),
    (1536, 12288),
)


def warmup_sm120_bf16_gemv(m: int, n: int, k: int) -> None:
    """Compile, load, and cudaFuncSetAttribute the (M, N, K) GEMV module.

    Must run OUTSIDE CUDA graph capture. A shape's first-ever run() is also a
    valid warmup, so SGLang's eager warmup forward before decode-graph
    capture covers every captured shape; for shapes captured with no prior
    eager call, call this at model-init time.
    """
    if not use_sm120_bf16_gemv(m, n, k):
        raise ValueError(f"(m={m}, n={n}, k={k}) is outside the SM120 BF16 GEMV gate")
    _jit_sm120_bf16_gemv_module(n, k, m)


def prewarm_sm120_bf16_gemv_e2b(ms: tuple[int, ...] = (1, 2, 4)) -> None:
    """Warm up every gemma-4-E2B decode (M, N, K) module (outside capture)."""
    for m in ms:
        for n, k in _E2B_NK:
            warmup_sm120_bf16_gemv(m, n, k)
