import torch
import torch.nn.functional as F
import triton

from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


@triton.testing.perf_report(
        triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=["vllm-fp8", "torch-fp8"],
        line_names=["vllm-fp8", "torch-fp8"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="GB/s",
        plot_name="int8 scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider):
    M, N, K = batch_size, 8192, 21760
    a = torch.ones((M, K), device="cuda") * 5.0
    b = torch.ones((N, K), device="cuda") * 5.0
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a)
    b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b)
    b_fp8 = b_fp8.t()
    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm-fp8":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_mm(
                a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, torch.bfloat16
            ),
            quantiles=quantiles,
        )
    if provider == "torch-fp8":
        scale_a_2d = scale_a_fp8.float().unsqueeze(1)  # [M, 1]
        scale_b_2d = scale_b_fp8.float().unsqueeze(0)  # [1, N]
        try:
            out = torch.empty(
                (a_fp8.shape[0], b_fp8.shape[0]), device="cuda", dtype=torch.bfloat16
            )
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch._scaled_mm(
                    a_fp8,
                    b_fp8,
                    out=out,
                    out_dtype=torch.bfloat16,
                    scale_a=scale_a_2d,
                    scale_b=scale_b_2d,
                    use_fast_accum=True,
                ),
                quantiles=quantiles,
            )
        except RuntimeError as e:
            print("Error details:", e)
            raise
    gbps = lambda ms: (2 * M * N * K + M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path="bench_int8_res")