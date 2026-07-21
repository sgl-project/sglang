import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.minicpm_sala import (
    get_block_table_v1,
    get_block_table_v2,
    get_block_table_v3,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

_HEAD_GROUP = 2
_SPARSE_BLOCK_SIZE = 64
_TOPK = 96

TOKEN_NUM_LIST = get_benchmark_range(
    full_range=[2**n for n in range(9, 15)],
    ci_range=[512, 4096],
)

configs = list(itertools.product(TOKEN_NUM_LIST))

_VERSION_FNS = {
    "v1": get_block_table_v1,
    "v2": get_block_table_v2,
    "v3": get_block_table_v3,
}


def _make_valid_inputs(token_num: int, topk: int, device: str = DEFAULT_DEVICE):
    """Well-formed inputs (only non-negative block indices) shared by v1/v2/v3.

    ``seqlen_q_max`` is tied to ``token_num`` so the per-token causal position
    (``token_pos_in_bs``) never indexes past ``block_table``.
    """
    seqlen_q_max = token_num
    num_blocks = max(1, seqlen_q_max // _SPARSE_BLOCK_SIZE)
    torch.manual_seed(0)
    topk_idx = torch.randint(
        0, num_blocks, (_HEAD_GROUP, token_num, topk), dtype=torch.int32, device=device
    )
    block_table = torch.arange(
        1, seqlen_q_max + 1, dtype=torch.int32, device=device
    ).reshape(1, seqlen_q_max)
    token_to_bs = torch.zeros((token_num,), dtype=torch.int32, device=device)
    token_pos_in_bs = torch.arange(1, token_num + 1, dtype=torch.int32, device=device)
    seqlen_q = torch.tensor([seqlen_q_max], dtype=torch.int32, device=device)
    return topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q


def _bench_one(token_num: int, provider: str):
    inputs = _make_valid_inputs(token_num, _TOPK)
    fn = lambda: _VERSION_FNS[provider](*inputs)

    # Trigger JIT compilation + module caching before timing so it never
    # happens inside the CUDA graph capture done by run_benchmark.
    fn()
    torch.cuda.synchronize()

    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["token_num"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["v1", "v2", "v3"],
        line_names=["get_block_table_v1", "get_block_table_v2", "get_block_table_v3"],
        styles=[("blue", "-"), ("green", "-."), ("red", "--")],
        ylabel="us",
        plot_name="get-block-table-performance",
        args={},
    )
)
def benchmark(token_num: int, provider: str):
    return _bench_one(token_num, provider)


if __name__ == "__main__":
    # Print a plain-text table directly instead of benchmark.run(), which pulls
    # in matplotlib via triton's plotting path (not always available locally).
    header = f"{'token_num':>10} | {'v1 (us)':>12} {'v2 (us)':>12} {'v3 (us)':>12}"
    print(header)
    print("-" * len(header))
    for token_num in TOKEN_NUM_LIST:
        cells = []
        for provider in ("v1", "v2", "v3"):
            median_us, _, _ = _bench_one(token_num, provider)
            cells.append(f"{median_us:>12.3f}")
        print(f"{token_num:>10} | " + " ".join(cells))
