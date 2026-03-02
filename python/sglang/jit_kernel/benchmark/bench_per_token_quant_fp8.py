import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.per_token_quant_fp8 import (
    per_token_quant_fp8 as jit_per_token_quant_fp8,
)

try:
    from sgl_kernel import sgl_per_token_quant_fp8 as aot_per_token_quant_fp8

    AOT_AVAILABLE = True
except ImportError:
    aot_per_token_quant_fp8 = None
    AOT_AVAILABLE = False


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    m, k = 16, 2048
    input_tensor = torch.randn(m, k, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    output_q_jit = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=DEFAULT_DEVICE)
    output_s_jit = torch.empty(m, dtype=torch.float32, device=DEFAULT_DEVICE)
    output_q_aot = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=DEFAULT_DEVICE)
    output_s_aot = torch.empty(m, dtype=torch.float32, device=DEFAULT_DEVICE)
    jit_per_token_quant_fp8(input_tensor, output_q_jit, output_s_jit)
    aot_per_token_quant_fp8(input_tensor, output_q_aot, output_s_aot)
    torch.testing.assert_close(output_q_jit, output_q_aot, rtol=0, atol=0)
    torch.testing.assert_close(output_s_jit, output_s_aot, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


M_LIST = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    ci_range=[16, 128],
)

K_LIST = get_benchmark_range(
    full_range=[2048, 4096, 7168],
    ci_range=[2048, 4096],
)

configs = list(itertools.product(M_LIST, K_LIST))

line_vals = ["jit", "pytorch"]
line_names = ["SGL JIT Kernel", "PyTorch"]
styles = [("blue", "-"), ("red", "--")]

if AOT_AVAILABLE:
    line_vals.insert(1, "aot")
    line_names.insert(1, "SGL AOT Kernel")
    styles.insert(1, ("green", "-."))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "k"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="per-token-quant-fp8-performance",
        args={},
    )
)
def benchmark(m, k, provider):
    FP8_E4M3_MAX = 448.0
    input_tensor = torch.randn(m, k, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    output_q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=DEFAULT_DEVICE)
    output_s = torch.empty(m, dtype=torch.float32, device=DEFAULT_DEVICE)

    if provider == "jit":

        def fn():
            jit_per_token_quant_fp8(input_tensor, output_q, output_s)

    elif provider == "aot":

        def fn():
            aot_per_token_quant_fp8(input_tensor, output_q, output_s)

    else:  # pytorch

        def fn():
            absmax = input_tensor.float().abs().max(dim=-1, keepdim=True).values
            scale = absmax / FP8_E4M3_MAX
            output_s.copy_(scale.squeeze(-1))
            scale_inv = torch.where(
                scale == 0, torch.zeros_like(scale), 1.0 / scale
            )
            quantized = (input_tensor.float() * scale_inv).clamp(
                -FP8_E4M3_MAX, FP8_E4M3_MAX
            )
            output_q.copy_(quantized.to(output_q.dtype))

    return run_benchmark(fn)


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
