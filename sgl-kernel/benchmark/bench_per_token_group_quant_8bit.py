import itertools
import time
from functools import partial
from pathlib import Path

import torch
import triton

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit, create_per_token_group_quant_fp8_output_scale
from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


# TODO temp
num_tokens_range = [768]
hidden_dim_range = [1536, 7168, 18432] # For DeepSeek V3/R1
# num_tokens_range = [1, 4, 16, 64, 256, 768, 2048, 8192, 16384]
# hidden_dim_range = [1536, 7168, 18432] # For DeepSeek V3/R1
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
    itertools.product(num_tokens_range, hidden_dim_range, group_size_range, dst_dtype_range, flags_range)
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

    quantiles = [0.5, 0.2, 0.8]

    fn = {
        "triton": triton_per_token_group_quant_8bit,
        "sglang": sglang_per_token_group_quant_8bit,
    }[provider]
    bench_fn = lambda: fn(x=x.clone(), group_size=group_size, dst_dtype=dst_dtype, **flags)

    # TODO no need?
    num_repeat = 10
    repeated_bench_fn = lambda: [bench_fn() for _ in range(num_repeat)]

    ms, min_ms, max_ms = triton.testing.do_bench(repeated_bench_fn, quantiles=quantiles)

    postprocess_time = lambda t_ms: t_ms * 1000 / num_repeat
    return postprocess_time(ms), postprocess_time(max_ms), postprocess_time(min_ms)


if __name__ == "__main__":
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
        benchmark.run(print_data=True)

    trace_path = str(Path("/data/numa0/tom/temp_sglang_server2local/") / f"{time.time()}.trace.json.gz")
    prof.export_chrome_trace(trace_path)
