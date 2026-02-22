import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.cast import downcast_fp8
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

DEVICE = DEFAULT_DEVICE
DTYPE = torch.bfloat16


# ── Scalar ("before") kernel ───────────────────────────────────────────────────


@cache_once
def _jit_cast_scalar_module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "cast_scalar",
        *args,
        cuda_files=["elementwise/cast_scalar.cuh"],
        cuda_wrappers=[("downcast_fp8_scalar", f"downcast_fp8_scalar<{args}>")],
    )


def downcast_fp8_scalar(k, v, k_out, v_out, k_scale, v_scale, loc, mult=1, offset=0):
    module = _jit_cast_scalar_module(k.dtype)
    module.downcast_fp8_scalar(k, v, k_out, v_out, k_scale, v_scale, loc, mult, offset)


# ── V2 kernel (256 threads fixed, 2D grid scaling) ────────────────────────────


@cache_once
def _jit_cast_v2_module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "cast_v2",
        *args,
        cuda_files=["elementwise/cast_v2.cuh"],
        cuda_wrappers=[("downcast_fp8_v2", f"downcast_fp8_v2<{args}>")],
    )


def downcast_fp8_v2(k, v, k_out, v_out, k_scale, v_scale, loc, mult=1, offset=0):
    module = _jit_cast_v2_module(k.dtype)
    module.downcast_fp8_v2(k, v, k_out, v_out, k_scale, v_scale, loc, mult, offset)


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

LINE_VALS = ["scalar", "vec", "v2"]
LINE_NAMES = ["Scalar (before)", "Vectorized (tile::Memory)", "V2 (256 threads, 2D grid)"]
STYLES = [("blue", "--"), ("orange", "-"), ("green", "-.")]


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
        plot_name="downcast-fp8-comparison",
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

    if provider == "scalar":
        fn = lambda: downcast_fp8_scalar(k, v, k_out, v_out, k_scale, v_scale, loc)
    elif provider == "vec":
        fn = lambda: downcast_fp8(k, v, k_out, v_out, k_scale, v_scale, loc)
    else:
        fn = lambda: downcast_fp8_v2(k, v, k_out, v_out, k_scale, v_scale, loc)

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

    scalar_fn = lambda: downcast_fp8_scalar(k, v, k_out, v_out, k_scale, v_scale, loc)
    vec_fn = lambda: downcast_fp8(k, v, k_out, v_out, k_scale, v_scale, loc)
    v2_fn = lambda: downcast_fp8_v2(k, v, k_out, v_out, k_scale, v_scale, loc)

    scalar_ms, _, _ = triton.testing.do_bench_cudagraph(scalar_fn, quantiles=[0.5, 0.2, 0.8])
    vec_ms, _, _ = triton.testing.do_bench_cudagraph(vec_fn, quantiles=[0.5, 0.2, 0.8])
    v2_ms, _, _ = triton.testing.do_bench_cudagraph(v2_fn, quantiles=[0.5, 0.2, 0.8])

    def fmt(ms):
        return f"{ms*1000:6.2f}us {total_bytes/(ms*1e-3)/1e9:6.0f}GB/s"

    print(
        f"  sl={input_sl:5d}  h={head:2d}  d={dim:4d}"
        f"  |  scalar {fmt(scalar_ms)}"
        f"  |  vec {fmt(vec_ms)}"
        f"  |  v2 {fmt(v2_ms)}"
        f"  |  v2 speedup {vec_ms/v2_ms:.2f}x"
    )


def report_bandwidth():
    print(f"\n{'='*110}")
    print("  Scalar vs Vectorized vs V2 (256 threads, 2D grid)")
    print(f"  dtype={DTYPE}, device={DEVICE}")
    print(f"{'='*110}")
    for sl in [64, 256, 1024, 2048]:
        for h, d in [(8, 128), (32, 128), (8, 256), (32, 256)]:
            _report_bandwidth(sl, h, d, DTYPE)
    print()


if __name__ == "__main__":
    benchmark.run(print_data=True)
    report_bandwidth()
