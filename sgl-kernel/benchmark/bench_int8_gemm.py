import torch
import triton
from sgl_kernel import int8_scaled_mm
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=["vllm", "sgl-kernel"],
        line_names=["vllm", "sgl-kernel"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="GB/s",
        plot_name="int8 scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider):
    M, N, K = batch_size, 8192, 21760
    a = to_int8(torch.randn((M, K), device="cuda") * 5)
    b = to_int8(torch.randn((N, K), device="cuda").t() * 5)
    o = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    # bias = torch.zeros((N,), device="cuda", dtype=torch.bfloat16)

    bias = None
    quantiles = [0.5, 0.2, 0.8]
    if provider == "sgl-kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: int8_scaled_mm(o, a, b, scale_a, scale_b, bias), quantiles=quantiles
        )
    if provider == "vllm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, bias=bias),
            quantiles=quantiles,
        )
    gbps = lambda ms: (2 * M * N * K - M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path="bench_int8_res")
