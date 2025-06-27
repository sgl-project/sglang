import itertools

import torch
import triton

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


batch_size_range = [1, 2, 4, 8, 16, 32, 64]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
group_size_range = [128]  # For DeepSeek V3/R1
dst_dtype_range = [torch.int8, fp8_type_]
flags_range = [TODO]

configs = list(
    itertools.product(
        batch_size_range, seq_len_range, group_size_range, dst_dtype_range, flags_range
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size", "dst_dtype", "flags"],
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
def benchmark(batch_size, seq_len, group_size, dst_dtype, flags, provider):
    device = torch.device("cuda")
    hidden_dim = 7168

    x = torch.randn(
        batch_size * seq_len, hidden_dim, device=device, dtype=torch.float16
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_per_token_group_quant_8bit(x.clone(), group_size, dst_dtype, **flags)
    elif provider == "sglang":
        fn = lambda: sglang_per_token_group_quant_8bit(x.clone(), group_size, dst_dtype, **flags)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
