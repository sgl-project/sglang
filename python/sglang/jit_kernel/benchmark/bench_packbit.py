"""
Benchmark: segment_packbits JIT vs AOT (sgl_kernel)

Measures throughput (µs) across typical batch sizes and segment lengths.

Run:
    python python/sglang/jit_kernel/benchmark/bench_packbit.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.packbit import segment_packbits as segment_packbits_jit

try:
    from sgl_kernel import segment_packbits as segment_packbits_aot

    AOT_AVAILABLE = True
except ImportError:
    segment_packbits_aot = None
    AOT_AVAILABLE = False

DEVICE = "cuda"

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 4, 16, 64],
    ci_range=[4],
)

SEG_LEN_RANGE = get_benchmark_range(
    full_range=[64, 256, 1024, 4096],
    ci_range=[256],
)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def make_inputs(bs, seg_len):
    torch.manual_seed(42)
    input_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    output_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    for i in range(bs):
        input_indptr[i + 1] = input_indptr[i] + seg_len
        output_indptr[i + 1] = output_indptr[i] + (seg_len + 7) // 8

    total_in = input_indptr[-1].item()
    total_out = output_indptr[-1].item()
    x = torch.randint(0, 2, (total_in,), dtype=torch.bool, device=DEVICE)
    y = torch.zeros(total_out, dtype=torch.uint8, device=DEVICE)

    return x, input_indptr, output_indptr, y


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "seg_len"],
        x_vals=list(itertools.product(BATCH_SIZE_RANGE, SEG_LEN_RANGE)),
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=[("blue", "--"), ("orange", "-")][: len(LINE_VALS)],
        ylabel="us",
        plot_name="segment-packbits-performance",
        args={},
    )
)
def bench_segment_packbits(bs: int, seg_len: int, provider: str):
    x, input_indptr, output_indptr, y = make_inputs(bs, seg_len)
    backup = y.clone()

    if provider == "jit":

        def fn():
            y.copy_(backup)
            segment_packbits_jit(x, input_indptr, output_indptr, y, batch_size=bs)

    elif provider == "aot":

        def fn():
            y.copy_(backup)
            segment_packbits_aot(x, input_indptr, output_indptr, y, batch_size=bs)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Quick correctness diff
# ---------------------------------------------------------------------------


def calculate_diff():
    if not AOT_AVAILABLE:
        print("sgl_kernel not available — skipping AOT diff check")
        return

    print("Correctness diff — segment_packbits (JIT vs AOT):")
    for bs, seg_len in [(1, 64), (4, 256), (8, 1024)]:
        x, input_indptr, output_indptr, y_jit = make_inputs(bs, seg_len)
        y_aot = y_jit.clone()

        segment_packbits_jit(x, input_indptr, output_indptr, y_jit, batch_size=bs)
        segment_packbits_aot(x, input_indptr, output_indptr, y_aot, batch_size=bs)

        status = "OK" if torch.equal(y_jit, y_aot) else "MISMATCH"
        print(f"  bs={bs:2d} seg_len={seg_len:5d}  [{status}]")


if __name__ == "__main__":
    calculate_diff()
    print()
    bench_segment_packbits.run(print_data=True)
