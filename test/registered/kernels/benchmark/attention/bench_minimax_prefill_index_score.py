"""Benchmark: MiniMax-M3 score-only prefill indexer, one split vs the split-KV launch.

Score + top-k for one request over a cached prefix, one index head per rank (TP4).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HD, BLK, TOPK, PAGE = 128, 128, 16, 128
SPLITS = {"one_split": 1, "split_kv": None}


def _index_step(
    q, k_cache, req_to_token, slot_ids, cu_seqlens, seq_lens, prefix_lens, **static
):
    return flash_prefill_with_topk_index(
        q=q,
        k_cache=k_cache,
        v_cache=None,
        sink=None,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        **static,
    )


@marker.parametrize(
    "prefix,new",
    [(80_512, 305), (124_928, 773), (250_000, 1_536), (200_000, 16_384)],
    [(80_512, 305), (200_000, 16_384)],
)
@marker.benchmark("impl", list(SPLITS))
def benchmark(prefix: int, new: int, impl: str):
    width = prefix + new
    pages = (width + PAGE - 1) // PAGE
    slots = torch.randperm(pages, device="cuda")[:, None] * PAGE + torch.arange(
        PAGE, device="cuda"
    )
    cu_seqlens = torch.tensor([0, new], dtype=torch.int32, device="cuda")
    args = (
        torch.randn(new, 1, HD, dtype=torch.bfloat16, device="cuda"),
        torch.randn(pages * PAGE, 1, HD, dtype=torch.bfloat16, device="cuda"),
        slots.reshape(1, -1)[:, :width].to(torch.int32).contiguous(),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        cu_seqlens,
        torch.tensor([width], dtype=torch.int32, device="cuda"),
        torch.tensor([prefix], dtype=torch.int32, device="cuda"),
    )
    # Precomputed so the captured call never syncs on the host.
    cu_seqblocks_q, max_seqblock_q, all_seqblock_q, *_ = get_cu_seqblocks(
        cu_seqlens, new, 1, BLK, [new]
    )
    static = dict(
        max_seqlen_q=new,
        max_seqlen_k=width,
        block_size_q=1,
        block_size_k=BLK,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        score_type="max",
        disable_index_value=True,
        page_size=PAGE,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
        all_seqblock_q=all_seqblock_q,
        num_kv_splits=SPLITS[impl],
    )
    return marker.do_bench(
        _index_step,
        input_args=args,
        input_kwargs=static,
        graph_clone_kwargs=None,
        memory_args=(args[1],),
    )


if __name__ == "__main__":
    benchmark.run()
