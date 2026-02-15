import os

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.jit_kernel.dsv3_fused_a_gemm import dsv3_fused_a_gemm as jit_dsv3_fused_a_gemm

try:
    from sgl_kernel import dsv3_fused_a_gemm as aot_dsv3_fused_a_gemm

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# Fixed problem dimensions (DeepSeek V3 fused A projection)
HD_IN = 7168
HD_OUT = 2112


def _make_inputs(num_tokens):
    """Create test inputs: mat_a (row-major bf16), mat_b (column-major bf16)."""
    mat_a = torch.randn(num_tokens, HD_IN, dtype=torch.bfloat16, device="cuda")
    mat_b = (
        torch.randn(HD_IN, HD_OUT, dtype=torch.bfloat16, device="cuda")
        .t()
        .contiguous()
        .t()
    )
    return mat_a, mat_b


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    mat_a, mat_b = _make_inputs(8)
    out_jit = jit_dsv3_fused_a_gemm(mat_a, mat_b)
    out_aot = aot_dsv3_fused_a_gemm(mat_a, mat_b)
    torch.testing.assert_close(out_jit, out_aot, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


if IS_CI:
    m_range = [1, 8, 16]
else:
    m_range = [i + 1 for i in range(16)]

if AOT_AVAILABLE:
    line_vals = ["jit", "aot", "torch"]
    line_names = ["JIT Kernel", "AOT Kernel", "torch (bf16)"]
    styles = [("blue", "-"), ("green", "-"), ("orange", "--")]
else:
    line_vals = ["jit", "torch"]
    line_names = ["JIT Kernel", "torch (bf16)"]
    styles = [("blue", "-"), ("orange", "--")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=m_range,
        x_log=False,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="TFLOPs",
        plot_name="dsv3-fused-a-gemm-performance",
        args={},
    )
)
def benchmark(num_tokens, provider):
    mat_a, mat_b = _make_inputs(num_tokens)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "jit":
        fn = lambda: jit_dsv3_fused_a_gemm(mat_a, mat_b)
    elif provider == "aot":
        fn = lambda: aot_dsv3_fused_a_gemm(mat_a, mat_b)
    elif provider == "torch":
        fn = lambda: F.linear(mat_a, mat_b.T)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * num_tokens * HD_IN * HD_OUT
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
