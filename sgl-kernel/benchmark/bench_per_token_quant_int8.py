import argparse
from typing import Tuple
import torch
import triton
from vllm._custom_ops import scaled_int8_quant as vllm_scaled_int8_quant

from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from sgl_kernel import sgl_per_token_quant_int8


def _test_accuracy_once(M, K, input_dtype, device):
    x = torch.randn(M, K, dtype=input_dtype, device=device) * 5000
    out, scales, _ = vllm_scaled_int8_quant(x, symmetric=True)
    out1, scales1 = per_token_quant_int8(x)
    out2, scales2 = sglang_per_token_quant_int8(x)
    torch.testing.assert_close(out, out1, atol=1, rtol=0)
    torch.testing.assert_close(out, out2, atol=1, rtol=0)
    torch.testing.assert_close(scales, scales2)
    torch.testing.assert_close(scales1, scales2)
    print(f"M: {M}, K: {K}, type: {input_dtype} OK")


def test_accuracy():
    Ms = [8, 16, 128, 1024, 2048, 4096]
    Ks = [512, 1024, 2048, 8192]
    input_dtypes = [torch.float16, torch.bfloat16]
    for M in Ms:
        for K in Ks:
            for input_dtype in input_dtypes:
                _test_accuracy_once(M, K, input_dtype, "cuda")

def sglang_per_token_quant_int8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scales = torch.empty(input.shape[:-1] + (1,), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=torch.int8)
    sgl_per_token_quant_int8(input, output, scales)

    return output, scales

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=["vllm op", "triton", "kernel"],
        line_names=["vllm op", "triton", "kernel"],
        styles=[("blue", "-"), ("orange", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="int8 per token quant",
        args={},
    )
)
def benchmark(batch_size, provider):
    M, K = batch_size, 16384
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 1000

    quantiles = [0.5, 0.2, 0.8]
    if provider == "vllm op":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_int8_quant(x, symmetric=True),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: per_token_quant_int8(x),
            quantiles=quantiles,
        )
    if provider == "kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sglang_per_token_quant_int8(x),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./bench_int8_quant_res",
        help="Path to save int8 quant benchmark results",
    )
    args = parser.parse_args()

    test_accuracy()

    benchmark.run(print_data=True, show_plots=True, save_path=args.save_path)
