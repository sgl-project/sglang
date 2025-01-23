import torch
import torch.nn.functional as F
import triton
from sgl_kernel import fp8_scaled_mm as sgl_scaled_mm
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=[
            "vllm-fp8-fp16",
            "vllm-fp8-bf16",
            "sglang-fp8-fp16",
            "sglang-fp8-bf16",
        ],
        line_names=[
            "vllm-fp8-fp16",
            "vllm-fp8-bf16",
            "sglang-fp8-fp16",
            "sglang-fp8-bf16",
        ],
        styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")],
        ylabel="GB/s",
        plot_name="fp8 scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider):
    M, N, K = batch_size, 4096, 8192
    a = torch.ones((M, K), device="cuda") * 5.0
    b = torch.ones((N, K), device="cuda") * 5.0
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a)
    b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b)
    b_fp8 = b_fp8.t()
    quantiles = [0.5, 0.2, 0.8]

    dtype = torch.float16 if "fp16" in provider else torch.bfloat16

    if "vllm-fp8" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype),
            quantiles=quantiles,
        )
    elif "sglang-fp8" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sgl_scaled_mm(
                a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=None
            ),
            quantiles=quantiles,
        )

    gbps = lambda ms: (2 * M * N * K + M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path="bench_fp8_res")
