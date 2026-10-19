"""Benchmark for the DeepSeek-V4 indexer top-k transform, with and without sorted output."""

import itertools

import torch
import triton
import triton.testing
from sgl_kernel import deepseek_v4_topk_transform_512

try:
    from sglang.utils import is_in_ci

    IS_CI = is_in_ci()
except ImportError:
    IS_CI = False

TOPK = 512
PAGE_SIZE = 64

batch_sizes = [1] if IS_CI else [1, 8, 32, 128]
seq_lens = [4096] if IS_CI else [1024, 4096, 32768]

configs = list(itertools.product(batch_sizes, seq_lens))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["unsorted", "sorted", "torch_sort"],
        line_names=["SGL Kernel", "SGL Kernel sort_output", "SGL Kernel + torch.sort"],
        styles=[("blue", "-"), ("green", "-"), ("red", "--")],
        ylabel="µs (median)",
        plot_name="dsv4-topk-transform-performance",
        args={},
    )
)
def benchmark_topk_transform(batch_size, seq_len, provider):
    torch.manual_seed(42)
    scores = torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    num_pages = triton.cdiv(seq_len, PAGE_SIZE)
    page_table = torch.stack(
        [
            torch.randperm(4 * num_pages, device="cuda")[:num_pages]
            for _ in range(batch_size)
        ]
    ).int()
    page_indices = torch.empty(batch_size, TOPK, dtype=torch.int32, device="cuda")

    def topk(sort_output):
        deepseek_v4_topk_transform_512(
            scores,
            lengths,
            page_table,
            page_indices,
            PAGE_SIZE,
            sort_output=sort_output,
        )

    if provider == "unsorted":
        fn = lambda: topk(False)
    elif provider == "sorted":
        fn = lambda: topk(True)
    else:
        fn = lambda: (topk(False), torch.sort(page_indices, dim=-1))

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark_topk_transform.run(print_data=True)
