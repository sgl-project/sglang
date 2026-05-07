"""
Microbenchmark: triton_fused_store_cache vs. the two-step AMD fallback.

Measures wall-clock µs per call at various batch sizes (num_tokens) for both
the flashmla and indexer cache variants.

Usage (inside the rocm/sgl-dev container, from the sglang repo root):
    python python/sglang/jit_kernel/benchmark/bench_triton_store_cache.py

Output: a Markdown table printed to stdout + written to
    /tmp/bench_triton_store_cache.md

Environment requirements:
    - ROCm / HIP GPU (MI300X / MI350X)
    - sglang installed editable (pip install -e python)
    - triton available (already present in the rocm/sgl-dev image)
"""
from __future__ import annotations

import math
import time
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# Guard: skip cleanly on CUDA machines
# ---------------------------------------------------------------------------

try:
    from sglang.srt.utils import is_hip

    _IS_HIP = is_hip()
except ImportError:
    _IS_HIP = False

if not torch.cuda.is_available():
    raise SystemExit("No GPU found — aborting benchmark.")
if not _IS_HIP:
    raise SystemExit("This benchmark is ROCm-only. Skipping on CUDA.")

from sglang.jit_kernel.triton_store_cache import (
    triton_fused_store_flashmla,
    triton_fused_store_indexer,
)
from sglang.srt.layers.attention.nsa.index_buf_accessor_v4 import (
    _set_k_and_s_triton,
)
from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_PAGE_SIZE = 256
_WARMUP = 200
_ITERS = 200

# ---------------------------------------------------------------------------
# Cache allocators
# ---------------------------------------------------------------------------


def _flashmla_bytes_per_page(page_size: int) -> int:
    return math.ceil(584 * page_size / 576) * 576


def _make_flashmla_cache(num_tokens: int, page_size: int) -> torch.Tensor:
    num_pages = num_tokens // page_size + 2
    bpp = _flashmla_bytes_per_page(page_size)
    return torch.zeros(num_pages, bpp, dtype=torch.uint8, device="cuda")


def _make_indexer_cache(num_tokens: int, page_size: int) -> torch.Tensor:
    num_pages = num_tokens // page_size + 2
    return torch.zeros(num_pages, 132 * page_size, dtype=torch.uint8, device="cuda")


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _bench(fn: Callable, warmup: int = _WARMUP, iters: int = _ITERS) -> float:
    """Return median µs per call over `iters` iterations after `warmup` warmup calls."""
    # Warmup (triggers Triton JIT compilation on first call)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Benchmark: flashmla
# ---------------------------------------------------------------------------


def bench_flashmla(num_tokens: int, page_size: int = _PAGE_SIZE):
    device = torch.device("cuda")
    key = torch.randn((num_tokens, 512), device=device, dtype=torch.bfloat16)
    loc = torch.randint(0, num_tokens, (num_tokens,), device=device, dtype=torch.int64)
    cache_fused = _make_flashmla_cache(num_tokens, page_size)
    cache_ref = _make_flashmla_cache(num_tokens, page_size)

    def run_fused():
        triton_fused_store_flashmla(key, cache_fused, loc, page_size)

    def run_ref():
        pack = quant_to_nope_fp8_rope_bf16_pack_triton(key)
        _set_k_and_s_triton(cache_ref, loc, pack, page_size)

    us_fused = _bench(run_fused)
    us_ref = _bench(run_ref)
    return us_fused, us_ref


# ---------------------------------------------------------------------------
# Benchmark: indexer
# ---------------------------------------------------------------------------


def bench_indexer(num_tokens: int, page_size: int = _PAGE_SIZE):
    device = torch.device("cuda")
    key = torch.randn((num_tokens, 128), device=device, dtype=torch.bfloat16)
    loc = torch.randint(0, num_tokens, (num_tokens,), device=device, dtype=torch.int64)
    cache_fused = _make_indexer_cache(num_tokens, page_size)

    # The two-step indexer fallback path calls act_quant (Triton block-wise
    # quantisation from nsa/triton_kernel.py) + set_index_k_scale_buffer scatter.
    # We baseline against act_quant-only (conservative — real fallback also pays
    # the paged-scatter overhead on top of this).
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    def run_fused():
        triton_fused_store_indexer(key, cache_fused, loc, page_size)

    def run_ref():
        # Quant-only baseline: same Triton quantise the fallback uses, no scatter.
        act_quant(key.view(-1, 128))

    us_fused = _bench(run_fused)
    us_ref = _bench(run_ref)
    return us_fused, us_ref


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]


def main():
    device_name = torch.cuda.get_device_name(0)
    print(f"\nDevice: {device_name}")
    print(f"Warmup: {_WARMUP} iters  |  Measured: {_ITERS} iters (median)\n")

    rows_flashmla = []
    rows_indexer = []

    print("Running flashmla benchmarks...")
    for n in BATCH_SIZES:
        us_f, us_r = bench_flashmla(n)
        speedup = us_r / us_f
        rows_flashmla.append((n, us_f, us_r, speedup))
        print(f"  N={n:>4}  fused={us_f:6.1f} µs  two-step={us_r:6.1f} µs  speedup={speedup:.2f}×")

    print("\nRunning indexer benchmarks...")
    for n in BATCH_SIZES:
        us_f, us_r = bench_indexer(n)
        speedup = us_r / us_f
        rows_indexer.append((n, us_f, us_r, speedup))
        print(f"  N={n:>4}  fused={us_f:6.1f} µs  ref-quant={us_r:6.1f} µs  speedup={speedup:.2f}×")

    # --- Build markdown tables ---
    md = []
    md.append(f"## Triton Fused Store Cache — {device_name} microbenchmark\n")
    md.append(f"**Device:** {device_name}  |  ROCm {torch.version.hip}  |  page_size={_PAGE_SIZE}")
    md.append(f"**Iterations:** {_ITERS} measured (median µs), {_WARMUP} warmup\n")

    md.append("### flashmla (input [N, 512] BF16 → paged SWA KV cache)")
    md.append("| num_tokens | fused (µs) | two-step (µs) | speedup |")
    md.append("| ---------- | ---------- | ------------- | ------- |")
    for n, uf, ur, sp in rows_flashmla:
        md.append(f"| {n} | {uf:.1f} | {ur:.1f} | {sp:.2f}× |")

    md.append("")
    md.append("### indexer (input [N, 128] BF16 → paged C4 indexer KV cache)")
    md.append("| num_tokens | fused (µs) | ref-quant (µs) | speedup |")
    md.append("| ---------- | ---------- | -------------- | ------- |")
    for n, uf, ur, sp in rows_indexer:
        md.append(f"| {n} | {uf:.1f} | {ur:.1f} | {sp:.2f}× |")

    output = "\n".join(md)
    out_path = "/tmp/bench_triton_store_cache.md"
    with open(out_path, "w") as f:
        f.write(output + "\n")

    print(f"\n{'='*60}")
    print(output)
    print(f"{'='*60}")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
