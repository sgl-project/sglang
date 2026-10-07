"""Indexer CP must select exactly the blocks the native TP selector does."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")

BLOCK, TOPK, HEADS, DIM = 128, 16, 4, 128
INIT_BLOCKS, LOCAL_BLOCKS = 1, 2
NDT = 4


def _inputs(batch, max_len, dtype, device):
    torch.manual_seed(20260928)
    padded_len = (max_len + BLOCK - 1) // BLOCK * BLOCK
    # Request i owns table row batch + 2 * (batch - 1 - i) + 1: never row i, out of
    # order, with unused rows between. A kernel that addresses K by batch or group
    # index instead of through the slot table reads another row's keys.
    table_rows = 3 * batch
    nslots = table_rows * padded_len
    # Physical pages shuffled independently of logical block ownership, so a rank
    # reading blocks r, r+4, ... cannot accidentally read contiguous memory.
    pages = torch.randperm(nslots // 16, device=device, dtype=torch.int32)
    table = (
        (pages[:, None] * 16 + torch.arange(16, device=device))
        .reshape(table_rows, padded_len)
        .to(torch.int32)
    )
    cache = torch.randn((nslots, 1, DIM), device=device, dtype=torch.bfloat16).to(dtype)
    q = torch.randn((batch, HEADS, DIM), device=device, dtype=torch.bfloat16)
    lengths = torch.full((batch,), max_len, dtype=torch.int64, device=device)
    slots = (batch + 1 + 2 * torch.arange(batch, device=device)).flip(0)
    return q, cache, table, slots, lengths


def _verify_rows(reqs, prefix, dtype, device):
    """Chain verify as the backend funnel lays it out: each request's NDT draft rows
    consecutively, row i seeing prefix + i + 1 tokens."""
    _, cache, table, slots, _ = _inputs(reqs, prefix + NDT, dtype, device)
    slots = slots.repeat_interleave(NDT)
    lengths = (prefix + torch.arange(1, NDT + 1, device=device)).repeat(reqs)
    q = torch.randn((reqs * NDT, HEADS, DIM), device=device, dtype=torch.bfloat16)
    return q, cache, table, slots, lengths


def _native_topk(q_head, cache, table, slots, lengths, max_len, k_scale):
    from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
        flash_decode_with_topk_idx,
    )

    return flash_decode_with_topk_idx(
        q=q_head,
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
        k_scale=k_scale,
    )[1]


def _cp_topk(q, cache, table, slots, lengths, max_len, k_scale, packed_queries=1):
    """Every rank's shard, computed in one process; the all-gather is the only part
    a four-rank run adds, and it is the runtime's collective, not this kernel."""
    from sglang.kernels.ops.attention.minimax_sparse.decode.indexer_cp import (
        merge_candidates,
        score_local_blocks,
        select_local_candidates,
    )

    gathered_q = q.permute(1, 0, 2).contiguous()
    keys = []
    for rank in range(HEADS):
        scores = score_local_blocks(
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
            k_scale,
            packed_queries=packed_queries,
        )
        keys.append(select_local_candidates(scores, lengths, rank, max_len, TOPK))
    gathered_keys = torch.stack(keys, dim=0)
    return [merge_candidates(gathered_keys, head) for head in range(HEADS)]


@unittest.skipUnless(torch.version.hip, "MiniMax indexer CP is gfx950-only")
class TestMiniMaxIndexerCP(unittest.TestCase):
    def test_sharded_selection_matches_the_native_selector(self):
        device = torch.device("cuda")
        for dtype in (torch.bfloat16, torch.float8_e4m3fnuz):
            for batch, max_len in ((1, 8192), (4, 32768)):
                with self.subTest(dtype=dtype, batch=batch, max_len=max_len):
                    q, cache, table, slots, lengths = _inputs(
                        batch, max_len, dtype, device
                    )
                    k_scale = 1.0
                    cp_heads = _cp_topk(
                        q, cache, table, slots, lengths, max_len, k_scale
                    )
                    for head in range(HEADS):
                        native = _native_topk(
                            q[:, head : head + 1],
                            cache,
                            table,
                            slots,
                            lengths,
                            max_len,
                            k_scale,
                        )
                        self.assertTrue(
                            torch.equal(cp_heads[head], native),
                            f"head {head}: {(cp_heads[head] != native).sum()} IDs differ",
                        )

    def test_packed_verify_rows_match_the_native_selector(self):
        """Packing a request's draft rows into one tile must select what scoring each
        row alone does, including rows whose lengths straddle a block boundary."""
        device = torch.device("cuda")
        prefix = 255 * BLOCK + 126  # rows end at 32767..32770, across block 256
        max_len = prefix + NDT
        for dtype in (torch.bfloat16, torch.float8_e4m3fnuz):
            with self.subTest(dtype=dtype):
                q, cache, table, slots, lengths = _verify_rows(4, prefix, dtype, device)
                packed = _cp_topk(
                    q, cache, table, slots, lengths, max_len, 1.0, packed_queries=NDT
                )
                for head in range(HEADS):
                    native = _native_topk(
                        q[:, head : head + 1],
                        cache,
                        table,
                        slots,
                        lengths,
                        max_len,
                        1.0,
                    )
                    self.assertTrue(
                        torch.equal(packed[head], native),
                        f"head {head}: {(packed[head] != native).sum()} IDs differ",
                    )


if __name__ == "__main__":
    unittest.main()
