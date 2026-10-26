"""On SM90, indexer CP must select exactly the blocks the native page-128 selector does.

Hopper serves the page-128 index cache through its own paged scorer, and at TP8 each
index head is served by two ranks, so eight shards score four distinct queries.
"""

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.indexer_cp import (
    merge_candidates,
    score_local_blocks,
    select_local_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK, TOPK, HEADS, DIM, PAGE = 128, 16, 4, 128, 128
INIT_BLOCKS, LOCAL_BLOCKS = 0, 1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the page-128 paged scorer is the SM90 decode path",
)


def _inputs(batch, max_len, dtype):
    gen = torch.Generator(device="cuda").manual_seed(20261025)
    # Spare rows so slots are never the batch index; pages shuffled so a rank's
    # strided blocks never land on contiguous memory.
    rows = 3 * batch
    pages = torch.randperm(rows * max_len // PAGE, generator=gen, device="cuda")
    table = (
        (pages.to(torch.int32)[:, None] * PAGE + torch.arange(PAGE, device="cuda"))
        .reshape(rows, max_len)
        .to(torch.int32)
    )
    cache = torch.randn(
        (rows * max_len, 1, DIM), generator=gen, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    q = torch.randn(
        (batch, HEADS, DIM), generator=gen, device="cuda", dtype=torch.bfloat16
    )
    slots = (batch + 1 + 2 * torch.arange(batch, device="cuda")).flip(0)
    # Off block boundaries, so the partial last block and the local block both count.
    lengths = max_len - (37 * torch.arange(batch, device="cuda")) % 200
    return q, cache, table, slots, lengths


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sharded_selection_matches_paged_native_selector(world, dtype):
    batch, max_len = 4, 32768
    q, cache, table, slots, lengths = _inputs(batch, max_len, dtype)
    head_of_rank = [rank * HEADS // world for rank in range(world)]
    gathered_q = q[:, head_of_rank].permute(1, 0, 2).contiguous()
    keys = torch.stack(
        [
            select_local_candidates(
                score_local_blocks(
                    gathered_q,
                    cache,
                    table,
                    slots,
                    lengths,
                    max_len,
                    rank,
                    INIT_BLOCKS,
                    LOCAL_BLOCKS,
                    DIM**-0.5,
                    1.0,
                ),
                lengths,
                rank,
                max_len,
                TOPK,
            )
            for rank in range(world)
        ]
    )
    for rank, head in enumerate(head_of_rank):
        native = flash_decode_with_topk_idx(
            q=q[:, head : head + 1].contiguous(),
            sink=None,
            k_cache=cache,
            v_cache=None,
            req_to_token=table,
            seq_lens=lengths,
            max_seqlen=max_len,
            slot_ids=slots,
            block_size=BLOCK,
            topk=TOPK,
            init_blocks=INIT_BLOCKS,
            local_blocks=LOCAL_BLOCKS,
            disable_index_value=True,
            page_size=PAGE,
        )[1]
        sharded = merge_candidates(keys, rank)
        assert torch.equal(sharded, native), (
            f"rank {rank}: {(sharded != native).any(-1).sum()} rows differ"
        )


@pytest.mark.parametrize("world", [4, 8])
def test_chain_verify_rows_match_paged_native_selector(world):
    """Verify rows straddle a block boundary; W=4 scores them packed, W=8 row by row."""
    reqs, draft_tokens = 4, 4
    prefix = 255 * BLOCK + 126  # rows end at 32767..32770, across block 256
    max_len = (prefix + draft_tokens + PAGE - 1) // PAGE * PAGE
    q, cache, table, slots, _ = _inputs(reqs * draft_tokens, max_len, torch.bfloat16)
    slots = slots[:reqs].repeat_interleave(draft_tokens)
    lengths = (prefix + torch.arange(1, draft_tokens + 1, device="cuda")).repeat(reqs)
    head_of_rank = [rank * HEADS // world for rank in range(world)]
    gathered_q = q[:, head_of_rank].permute(1, 0, 2).contiguous()
    keys = torch.stack(
        [
            select_local_candidates(
                score_local_blocks(
                    gathered_q,
                    cache,
                    table,
                    slots,
                    lengths,
                    max_len,
                    rank,
                    INIT_BLOCKS,
                    LOCAL_BLOCKS,
                    DIM**-0.5,
                    1.0,
                    packed_queries=draft_tokens,
                ),
                lengths,
                rank,
                max_len,
                TOPK,
            )
            for rank in range(world)
        ]
    )
    for rank, head in enumerate(head_of_rank):
        native = flash_decode_with_topk_idx(
            q=q[:, head : head + 1].contiguous(),
            sink=None,
            k_cache=cache,
            v_cache=None,
            req_to_token=table,
            seq_lens=lengths,
            max_seqlen=max_len,
            slot_ids=slots,
            block_size=BLOCK,
            topk=TOPK,
            init_blocks=INIT_BLOCKS,
            local_blocks=LOCAL_BLOCKS,
            disable_index_value=True,
            page_size=PAGE,
        )[1]
        sharded = merge_candidates(keys, rank)
        assert torch.equal(sharded, native), (
            f"rank {rank}: {(sharded != native).any(-1).sum()} rows differ"
        )
