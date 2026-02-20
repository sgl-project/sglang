import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm as jit_dsv3_router_gemm

try:
    from sgl_kernel import dsv3_router_gemm as aot_dsv3_router_gemm

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

HIDDEN_DIM = 7168
NUM_TOKENS = [i + 1 for i in range(16)]


if AOT_AVAILABLE:
    LINE_VALS = ["jit-256", "aot-256", "torch-256", "jit-384", "aot-384", "torch-384"]
    LINE_NAMES = ["JIT-256", "AOT-256", "torch-256", "JIT-384", "AOT-384", "torch-384"]
    STYLES = [
        ("blue", "-"),
        ("green", "-"),
        ("orange", "--"),
        ("blue", ":"),
        ("green", ":"),
        ("orange", "-."),
    ]
else:
    LINE_VALS = ["jit-256", "torch-256", "jit-384", "torch-384"]
    LINE_NAMES = ["JIT-256", "torch-256", "JIT-384", "torch-384"]
    STYLES = [("blue", "-"), ("orange", "--"), ("blue", ":"), ("orange", "-.")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=NUM_TOKENS,
        x_log=False,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="TFLOPs",
        plot_name="dsv3-router-gemm-bf16-out",
        args={"out_dtype": torch.bfloat16},
    )
)
def benchmark_bf16(num_tokens, provider, out_dtype):
    num_experts = 256 if provider.endswith("256") else 384

    mat_a = torch.randn(num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(num_experts, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")

    if provider.startswith("jit"):
        fn = lambda: jit_dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)
    elif provider.startswith("aot"):
        fn = lambda: aot_dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)
    elif provider.startswith("torch"):
        fn = lambda: F.linear(mat_a, mat_b)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )

    def tflops(t_ms):
        flops = 2 * num_tokens * HIDDEN_DIM * num_experts
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=NUM_TOKENS,
        x_log=False,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="TFLOPs",
        plot_name="dsv3-router-gemm-fp32-out",
        args={"out_dtype": torch.float32},
    )
)
def benchmark_fp32(num_tokens, provider, out_dtype):
    num_experts = 256 if provider.endswith("256") else 384

    mat_a = torch.randn(num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(num_experts, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")

    if provider.startswith("jit"):
        fn = lambda: jit_dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)
    elif provider.startswith("aot"):
        fn = lambda: aot_dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)
    elif provider.startswith("torch"):
        fn = lambda: F.linear(mat_a, mat_b).float()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )

    def tflops(t_ms):
        flops = 2 * num_tokens * HIDDEN_DIM * num_experts
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    benchmark_bf16.run(print_data=True)
    benchmark_fp32.run(print_data=True)
