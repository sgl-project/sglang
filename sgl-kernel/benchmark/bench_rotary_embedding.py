import itertools

import torch
import triton

from sglang.srt.bench_utils import bench_kineto

configs = [
    (batch_size, seq_len, save_kv_cache)
    for save_kv_cache in (False, True)
    for batch_size, seq_len in (
        (1, 1),
        (32, 1),
        (2, 512),
    )
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "save_kv_cache"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="us",
        args={},
    )
)
def benchmark(batch_size, seq_len, save_kv_cache, provider):
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
