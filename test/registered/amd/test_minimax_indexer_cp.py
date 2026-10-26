"""Indexer CP must select exactly the blocks the native TP selector does."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

BLOCK, TOPK, HEADS, DIM = 128, 16, 4, 128
INIT_BLOCKS, LOCAL_BLOCKS = 1, 2
DRAFT_TOKENS = 4


def _inputs(batch, max_len, dtype, device):
    torch.manual_seed(20260928)
    padded_len = (max_len + BLOCK - 1) // BLOCK * BLOCK
    # slots are never the row or group index, so addressing K by index reads the wrong keys
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
    """Chain verify as the backend funnel lays it out: each request's DRAFT_TOKENS draft rows
    consecutively, row i seeing prefix + i + 1 tokens."""
    _, cache, table, slots, _ = _inputs(reqs, prefix + DRAFT_TOKENS, dtype, device)
    slots = slots.repeat_interleave(DRAFT_TOKENS)
    lengths = (prefix + torch.arange(1, DRAFT_TOKENS + 1, device=device)).repeat(reqs)
    q = torch.randn(
        (reqs * DRAFT_TOKENS, HEADS, DIM), device=device, dtype=torch.bfloat16
    )
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


def _nvfp4_verify_inputs(prefixes, num_draft, seed, tied_pages=False):
    """DSpark verify rows over the packed NVFP4 index-K pool: whole 128-token pages,
    shuffled. With ``tied_pages`` most pages copy page 1, so most blocks tie."""
    from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
    from sglang.kernels.ops.quantization.nvfp4_kv_triton import nvfp4_kv_quantize

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    max_len = max(prefixes) + num_draft
    pages_per_req = -(-max_len // BLOCK)
    num_pages = len(prefixes) * pages_per_req
    pages = torch.randperm(num_pages, device=device, generator=gen) + 1
    offs = torch.arange(BLOCK, device=device)
    table = (
        (pages.view(len(prefixes), pages_per_req, 1) * BLOCK + offs)
        .view(len(prefixes), -1)[:, :max_len]
        .to(torch.int32)
    )
    k = torch.randn((num_pages + 1) * BLOCK, DIM, device=device, generator=gen)
    if tied_pages:
        k_pages = k.view(num_pages + 1, BLOCK * DIM)
        rare = torch.arange(num_pages + 1, device=device) % 13 == 0
        k_pages.copy_(k_pages[1 + rare.long()])
    payload, scales = nvfp4_kv_quantize(
        k.to(torch.bfloat16), torch.ones(1, device=device)
    )
    cache = Nvfp4Cache(
        payload=payload.view(num_pages + 1, BLOCK, DIM // 2),
        scales=scales.view(num_pages + 1, BLOCK, DIM // 16),
    )
    rows = len(prefixes) * num_draft
    q = torch.randn(rows, HEADS, DIM, device=device, generator=gen).to(torch.bfloat16)
    lengths = torch.tensor(
        [p + j + 1 for p in prefixes for j in range(num_draft)],
        dtype=torch.int64,
        device=device,
    )
    slots = torch.arange(len(prefixes), device=device).repeat_interleave(num_draft)
    # graph capture sizes the score columns past the real lengths
    return q, cache, table, slots, lengths, max_len + 4 * BLOCK


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "MiniMax indexer CP is gfx950-only"
)
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
        max_len = prefix + DRAFT_TOKENS
        for dtype in (torch.bfloat16, torch.float8_e4m3fnuz):
            with self.subTest(dtype=dtype):
                q, cache, table, slots, lengths = _verify_rows(4, prefix, dtype, device)
                packed = _cp_topk(
                    q,
                    cache,
                    table,
                    slots,
                    lengths,
                    max_len,
                    1.0,
                    packed_queries=DRAFT_TOKENS,
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

    def test_nvfp4_dspark_verify_rows_match_the_native_selector(self):
        """On M3.1's NVFP4 index-K pool, sharded verify scoring must select exactly
        what the native NVFP4 verify paths (fused, and score + radix top-k) do, for
        DSpark's 8-row and EAGLE's 4-row chains, short rows and tied scores."""
        from sglang.kernels.ops.attention.minimax_sparse.common.utils import (
            unit_scale,
        )
        from sglang.kernels.ops.attention.minimax_sparse.decode import (
            verify_topk_idx as verify_module,
        )
        from sglang.kernels.ops.attention.minimax_sparse.decode.verify_topk_gfx950 import (
            cp_verify_local_keys,
            cp_verify_merge,
        )

        sm_scale, k_scale = 0.07, 0.5
        for num_draft in (8, 4):
            for init_blocks, local_blocks in ((0, 1), (2, 2)):
                for num_reqs, tied, q_dtype in (
                    (1, False, torch.bfloat16),
                    (3, True, torch.bfloat16),
                    (17, False, torch.float16),
                    (64, True, torch.bfloat16),
                ):
                    prefixes = [
                        BLOCK * (60 + 37 * (r % 8)) - r for r in range(num_reqs)
                    ]
                    if num_reqs > 5:
                        prefixes[3], prefixes[5] = 3 * BLOCK - 2, 777  # <= top-k blocks
                    q, cache, table, slots, lengths, max_len = _nvfp4_verify_inputs(
                        prefixes, num_draft, seed=num_reqs, tied_pages=tied
                    )
                    q, q_scale = q.to(q_dtype), None
                    gathered_q = q.permute(1, 0, 2).contiguous()
                    keys = [
                        cp_verify_local_keys(
                            gathered_q,
                            cache,
                            table,
                            lengths,
                            max_len,
                            slots,
                            num_draft,
                            BLOCK,
                            init_blocks,
                            local_blocks,
                            sm_scale * unit_scale(q_scale) * unit_scale(k_scale),
                            HEADS,
                            rank,
                        )
                        for rank in range(HEADS)
                    ]
                    gathered_keys = torch.stack(keys, dim=0)
                    for fused in (True, False):
                        for head in range(HEADS):
                            with self.subTest(
                                num_draft=num_draft,
                                init=init_blocks,
                                local=local_blocks,
                                reqs=num_reqs,
                                tied=tied,
                                q=q_dtype,
                                fused=fused,
                                head=head,
                            ):
                                kwargs = dict(
                                    q=q[:, head : head + 1].contiguous(),
                                    k_cache=cache,
                                    req_to_token=table,
                                    seq_lens=lengths,
                                    max_seqlen=max_len,
                                    slot_ids=slots,
                                    num_draft=num_draft,
                                    block_size=BLOCK,
                                    topk=TOPK,
                                    init_blocks=init_blocks,
                                    local_blocks=local_blocks,
                                    sm_scale=sm_scale,
                                    q_scale=q_scale,
                                    k_scale=k_scale,
                                )
                                if fused:
                                    kwargs["page_size"] = BLOCK
                                    native = verify_module.verify_topk_idx(**kwargs)
                                else:
                                    use = verify_module._use_fused_verify_topk
                                    verify_module._use_fused_verify_topk = (
                                        lambda *a, **kw: False
                                    )
                                    try:
                                        native = verify_module.verify_topk_idx(**kwargs)
                                    finally:
                                        verify_module._use_fused_verify_topk = use
                                cp = cp_verify_merge(
                                    gathered_keys,
                                    head,
                                    lengths,
                                    max_len,
                                    table,
                                    BLOCK,
                                    TOPK,
                                    init_blocks,
                                    local_blocks,
                                )
                                self.assertTrue(
                                    torch.equal(cp, native),
                                    f"{(cp != native).sum()} IDs differ",
                                )


if __name__ == "__main__":
    unittest.main()
