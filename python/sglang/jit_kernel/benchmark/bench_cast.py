import torch
import triton
import triton.testing
from sgl_kernel import downcast_fp8 as downcast_fp8_aot

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.cast import downcast_fp8 as downcast_fp8_jit
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-benchmark-1-gpu-large")

DEVICE = DEFAULT_DEVICE
DTYPE = torch.bfloat16


# ── Config ranges ──────────────────────────────────────────────────────────────

SL_LIST = get_benchmark_range(
    full_range=[4, 16, 64, 256, 512, 1024, 2048],
    ci_range=[4, 64],
)

HEAD_DIM_LIST = get_benchmark_range(
    full_range=[(8, 128), (32, 128), (8, 256), (32, 256)],
    ci_range=[(8, 128)],
)

CONFIGS = [(sl, h, d, sl * 2) for sl in SL_LIST for h, d in HEAD_DIM_LIST]

LINE_VALS = ["aot", "jit"]
LINE_NAMES = ["AOT (sgl-kernel)", "JIT (cast.cuh, 256 threads, 2D grid)"]
STYLES = [("blue", "--"), ("orange", "-")]


# ── Perf report ────────────────────────────────────────────────────────────────


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["input_sl", "head", "dim", "out_sl"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="downcast-fp8-aot-vs-jit",
        args={},
    )
)
def benchmark(input_sl, head, dim, out_sl, provider):
    k = torch.randn(input_sl, head, dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(input_sl, head, dim, dtype=DTYPE, device=DEVICE)
    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device=DEVICE)
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device=DEVICE)
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    loc = torch.arange(input_sl, dtype=torch.int64, device=DEVICE)

    if provider == "aot":
        fn = lambda: downcast_fp8_aot(k, v, k_out, v_out, k_scale, v_scale, loc)
    else:
        fn = lambda: downcast_fp8_jit(k, v, k_out, v_out, k_scale, v_scale, loc)

    return run_benchmark(fn)


# ── Bandwidth analysis ─────────────────────────────────────────────────────────


def _report_bandwidth(input_sl, head, dim, dtype):
    elem_bytes = torch.finfo(dtype).bits // 8
    total_bytes = input_sl * head * dim * (2 * elem_bytes + 2)

    k = torch.randn(input_sl, head, dim, dtype=dtype, device=DEVICE)
    v = torch.randn(input_sl, head, dim, dtype=dtype, device=DEVICE)
    k_out = torch.zeros(input_sl * 2, head, dim, dtype=torch.uint8, device=DEVICE)
    v_out = torch.zeros(input_sl * 2, head, dim, dtype=torch.uint8, device=DEVICE)
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    loc = torch.arange(input_sl, dtype=torch.int64, device=DEVICE)

    aot_fn = lambda: downcast_fp8_aot(k, v, k_out, v_out, k_scale, v_scale, loc)
    jit_fn = lambda: downcast_fp8_jit(k, v, k_out, v_out, k_scale, v_scale, loc)

    aot_ms, _, _ = triton.testing.do_bench(aot_fn, quantiles=[0.5, 0.2, 0.8])
    jit_ms, _, _ = triton.testing.do_bench(jit_fn, quantiles=[0.5, 0.2, 0.8])

    def fmt(ms):
        return f"{ms*1000:6.2f}us {total_bytes/(ms*1e-3)/1e9:6.0f}GB/s"

    print(
        f"  sl={input_sl:5d}  h={head:2d}  d={dim:4d}"
        f"  |  aot {fmt(aot_ms)}"
        f"  |  jit {fmt(jit_ms)}"
        f"  |  speedup {aot_ms/jit_ms:.2f}x"
    )


def report_bandwidth():
    print(f"\n{'='*95}")
    print("  AOT (sgl-kernel) vs JIT (cast.cuh, 256 threads, 2D grid)")
    print(f"  dtype={DTYPE}, device={DEVICE}")
    print(f"{'='*95}")
    for sl in [64, 256, 1024, 2048]:
        for h, d in [(8, 128), (32, 128), (8, 256), (32, 256)]:
            _report_bandwidth(sl, h, d, DTYPE)
    print()


if __name__ == "__main__":
    benchmark.run(print_data=True)
    report_bandwidth()
