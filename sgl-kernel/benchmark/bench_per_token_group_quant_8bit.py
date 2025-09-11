import itertools
import time
from functools import partial
from pathlib import Path

import torch
import triton

from sglang.srt.bench_utils import bench_kineto
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


num_tokens_range = [1, 4, 16, 64, 256, 768, 2048, 8192, 16384]
hidden_dim_range = [1536, 7168, 18432]  # For DeepSeek V3/R1
group_size_range = [128]  # For DeepSeek V3/R1
# TODO test int8
dst_dtype_range = [fp8_type_]
flags_range = [
    dict(
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    ),
]


configs = list(
    itertools.product(
        num_tokens_range,
        hidden_dim_range,
        group_size_range,
        dst_dtype_range,
        flags_range,
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "hidden_dim", "group_size", "dst_dtype", "flags"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "sglang"],
        line_names=["Triton", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-group-quant-8bit-performance",
        args={},
    )
)
def benchmark(num_tokens, hidden_dim, group_size, dst_dtype, flags, provider):
    if flags["scale_ue8m0"] and group_size != 128:
        return

    device = torch.device("cuda")

    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)

    fn, kernel_names = {
        "triton": (triton_per_token_group_quant_8bit, "_per_token_group_quant_fp8"),
        "sglang": (
            sglang_per_token_group_quant_8bit,
            "per_token_group_quant_8bit_kernel",
        ),
    }[provider]
    bench_fn = lambda: fn(x=x, group_size=group_size, dst_dtype=dst_dtype, **flags)

    time_s = bench_kineto(bench_fn, kernel_names=kernel_names)
    return time_s * 1e6


if __name__ == "__main__":
    benchmark.run(print_data=True)
