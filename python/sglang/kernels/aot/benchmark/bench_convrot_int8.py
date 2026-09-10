import itertools

import torch
import triton
import triton.testing
from sgl_kernel import (
    convrot_int8_fused_linear,
    convrot_int8_linear_prequant,
    convrot_rotate_quantize_activation,
)

from sglang.utils import is_in_ci

IS_CI = is_in_ci()
GROUP_SIZE = 256

# Qwen-Image DiT linears at 1024x1024; same shapes as tests/test_convrot_int8.py.
M_VALUES = [2048] if IS_CI else [3, 20, 2048, 4096]
KN_VALUES = [(3072, 3072)] if IS_CI else [(3072, 3072), (3072, 12288), (12288, 3072)]
configs = [(M, K, N) for M, (K, N) in itertools.product(M_VALUES, KN_VALUES)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "K", "N"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["convrot_int8", "convrot_int8_prequant", "torch_bf16"],
        line_names=[
            "ConvRot INT8 (rotate + quant + GEMM)",
            "ConvRot INT8 (GEMM only)",
            "torch.addmm BF16",
        ],
        styles=[("green", "-"), ("green", "--"), ("red", "-")],
        ylabel="us (median)",
        plot_name="convrot-int8-linear",
        args={},
    )
)
def benchmark(M, K, N, provider):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    weight_q, weight_scale = convrot_rotate_quantize_activation(
        weight, group_size=GROUP_SIZE
    )
    x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)

    if provider == "convrot_int8":

        def fn():
            return convrot_int8_fused_linear(
                x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
            )

    elif provider == "convrot_int8_prequant":

        def fn():
            return convrot_int8_linear_prequant(
                x_q, x_scale, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
            )

    else:

        def fn():
            return torch.addmm(bias, x, weight.t())

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    if torch.cuda.get_device_capability() in ((9, 0), (10, 0)):
        benchmark.run(print_data=True)
    else:
        print("Skipping: convrot_int8 kernels are built for CC 9.0 and CC 10.0 only")
