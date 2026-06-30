import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.moe_align_block_size import (
    moe_align_block_size as jit_moe_align_block_size,
)

try:
    from sgl_kernel import moe_align_block_size as aot_moe_align_block_size

    AOT_AVAILABLE = True
except ImportError:
    aot_moe_align_block_size = None
    AOT_AVAILABLE = False


M_LIST = get_benchmark_range(
    full_range=[1, 4, 16, 64, 128, 256, 512, 1024],
    ci_range=[16, 128],
)

TOPK_LIST = get_benchmark_range(
    full_range=[2, 4, 8],
    ci_range=[2, 4],
)

NUM_EXPERTS_LIST = get_benchmark_range(
    full_range=[8, 64, 256],
    ci_range=[8, 64],
)

BLOCK_SIZE_LIST = get_benchmark_range(
    full_range=[64, 128],
    ci_range=[64],
)

configs = list(itertools.product(M_LIST, TOPK_LIST, NUM_EXPERTS_LIST, BLOCK_SIZE_LIST))

line_vals = ["jit"]
line_names = ["SGL JIT Kernel"]
styles = [("blue", "-")]

if AOT_AVAILABLE:
    line_vals.append("aot")
    line_names.append("SGL AOT Kernel")
    styles.append(("green", "-."))


def _make_args(m, topk, num_experts, block_size, device=DEFAULT_DEVICE):
    numel = m * topk
    ne_param = num_experts + 1

    topk_ids = torch.randint(0, num_experts, (numel,), dtype=torch.int32, device=device)

    if numel < ne_param + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + ne_param * (block_size - 1)

    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    return topk_ids, ne_param, block_size, max_num_tokens_padded, max_num_m_blocks


def _run_kernel(
    fn, topk_ids, ne_param, bs, max_tokens, max_blocks, device=DEFAULT_DEVICE
):
    sorted_ids = torch.empty((max_tokens,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    cumsum_buf = torch.empty((ne_param + 1,), dtype=torch.int32, device=device)
    fn(
        topk_ids,
        ne_param,
        bs,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buf,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    topk_ids, ne_param, bs, max_tokens, max_blocks = _make_args(64, 4, 64, 64)
    jit_sorted, jit_experts, jit_ntpp = _run_kernel(
        jit_moe_align_block_size,
        topk_ids,
        ne_param,
        bs,
        max_tokens,
        max_blocks,
    )
    aot_sorted, aot_experts, aot_ntpp = _run_kernel(
        aot_moe_align_block_size,
        topk_ids,
        ne_param,
        bs,
        max_tokens,
        max_blocks,
    )
    torch.testing.assert_close(jit_ntpp, aot_ntpp, rtol=0, atol=0)
    n = jit_ntpp.item()
    num_blocks = n // bs
    torch.testing.assert_close(
        jit_experts[:num_blocks], aot_experts[:num_blocks], rtol=0, atol=0
    )
    print("Correctness check passed (JIT vs AOT)")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "topk", "num_experts", "block_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(m, topk, num_experts, block_size, provider):
    topk_ids, ne_param, bs, max_tokens, max_blocks = _make_args(
        m, topk, num_experts, block_size
    )

    if provider == "jit":
        fn = jit_moe_align_block_size
    else:
        fn = aot_moe_align_block_size

    def bench_fn():
        _run_kernel(fn, topk_ids, ne_param, bs, max_tokens, max_blocks)

    return run_benchmark(bench_fn)


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
