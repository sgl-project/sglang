"""MiniMax-M3 sparse decode: a block inside one page loads from its first slot.

These cases pin that this matches the slot gather on the allocator's page layout,
that pages smaller than a block keep the gather, and that HiSparse slots win.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

BS, NQH, NKH, HD, BLK, TOPK = 4, 64, 1, 128, 128, 16
# A whole number of blocks; a partial last block goes through pos_mask on both paths.
SEQ_LEN = 1024
NUM_SLOTS = BS * SEQ_LEN


def _inputs(page_size=None):
    q = torch.randn(BS, NQH, HD, dtype=torch.bfloat16, device="cuda")
    # BLK spare rows: a one-tile read on a layout that breaks the precondition
    # may run up to a block past the last slot.
    k_cache = torch.randn(NUM_SLOTS + BLK, NKH, HD, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    req_to_token = torch.empty(BS, SEQ_LEN, dtype=torch.int32, device="cuda")
    for b in range(BS):
        if page_size is None:
            # A bare slot permutation: blocks no longer sit inside one page.
            slots = torch.randperm(SEQ_LEN, device="cuda") + b * SEQ_LEN
        else:
            # PagedTokenToKVPoolAllocator: page * page_size + offset, pages shuffled.
            npages = SEQ_LEN // page_size
            pages = torch.randperm(npages, device="cuda") + b * npages
            slots = pages[:, None] * page_size + torch.arange(page_size, device="cuda")
        req_to_token[b] = slots.reshape(-1).to(torch.int32)
    seq_lens = torch.full((BS,), SEQ_LEN, dtype=torch.int32, device="cuda")
    slot_ids = torch.arange(BS, dtype=torch.int64, device="cuda")
    num_blocks = SEQ_LEN // BLK
    topk_idx = torch.full((NKH, BS, TOPK), -1, dtype=torch.int32, device="cuda")
    for b in range(BS):
        chosen = torch.randperm(num_blocks, device="cuda")[:TOPK]
        topk_idx[0, b, : chosen.numel()] = chosen.to(torch.int32)
    return q, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx


def _decode(inputs, page_size, hisparse_slots=None):
    q, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx = inputs
    return flash_decode_with_gqa_share_sparse(
        q=q,
        sink=None,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        slot_ids=slot_ids,
        block_size=BLK,
        topk_idx=topk_idx,
        page_size=page_size,
        hisparse_slots=hisparse_slots,
    )


@pytest.mark.parametrize("page_size,engages", [(128, True), (256, True), (64, False)])
def test_paged_tile_matches_slot_gather(page_size, engages):
    """A block read as one tile matches the slot gather; small pages fall back."""
    torch.manual_seed(0)
    inputs = _inputs(page_size)
    gather = _decode(inputs, page_size=0)
    assert torch.equal(gather, _decode(inputs, page_size=page_size))
    if not engages:
        # These pages split every block, so a one-tile read differs;
        # the equality above therefore means the gather ran.
        assert not torch.equal(gather, _decode(inputs, page_size=BLK))
    else:
        # On a layout that breaks the precondition the tile must diverge,
        # else the equality above does not prove the tile ran.
        scattered = _inputs()
        assert not torch.equal(
            _decode(scattered, page_size=0), _decode(scattered, page_size=page_size)
        )


def test_hisparse_slots_override_paged_tile():
    """Pre-resolved HiSparse slots win over the paged tile."""
    torch.manual_seed(0)
    inputs = _inputs(BLK)
    _, _, _, req_to_token, _, _, topk_idx = inputs
    # Resolve every selected block through another slot permutation,
    # laid out like the coordinator's output: [1, batch, topk * block] int32.
    remap = torch.randperm(NUM_SLOTS, device="cuda").to(torch.int32)
    hisparse_slots = torch.full(
        (1, BS, TOPK * BLK), -1, dtype=torch.int32, device="cuda"
    )
    for b in range(BS):
        for t, block in enumerate(topk_idx[0, b].tolist()):
            if block >= 0:
                tokens = req_to_token[b, block * BLK : (block + 1) * BLK].long()
                hisparse_slots[0, b, t * BLK : (t + 1) * BLK] = remap[tokens]
    expected = _decode(inputs, page_size=0, hisparse_slots=hisparse_slots)
    assert torch.equal(
        expected, _decode(inputs, page_size=BLK, hisparse_slots=hisparse_slots)
    )
    # The remapped slots change the result,
    # else a tile that ignored them would pass the assertion above.
    assert not torch.equal(expected, _decode(inputs, page_size=BLK))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
