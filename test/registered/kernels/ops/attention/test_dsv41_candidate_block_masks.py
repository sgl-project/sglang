"""The fused level-one / level-two selection of the DeepSeek-V4.1 dense prefill
indexer must pick the same blocks and positions as the torch masks it replaces."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    mask_topk_scores,
    select_candidate_block_masks,
    select_candidate_blocks,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK = 8
TOPK_BLOCKS = 2048
TOPK = 512


def _case(rows, max_len, seed):
    torch.manual_seed(seed)
    width = (max_len + 7) // 8 * 8
    lens = torch.randint(1, max_len + 1, (rows,), device="cuda", dtype=torch.int32)
    lens[0] = 0  # a query that cannot see any compressed position yet
    lens[-1] = max_len
    # the indexer writes garbage past a row's length; make it loud
    logits = torch.randn(rows, width, device="cuda") * 3
    return logits, lens


def _reference_position_mask(logits, lens):
    j = torch.arange(logits.shape[1], device=logits.device)
    scores = logits.masked_fill(j[None, :] >= lens[:, None], -torch.inf)
    return select_candidate_blocks(
        scores, lens[:, None], topk_blocks=TOPK_BLOCKS, block_size=BLOCK
    )


def _fused_block_mask(logits, lens):
    blocks = (logits.shape[1] + BLOCK - 1) // BLOCK
    keep = torch.zeros(logits.shape[0], blocks + 1, dtype=torch.uint8, device="cuda")
    select_candidate_block_masks(
        logits, lens, topk_blocks=TOPK_BLOCKS, block_size=BLOCK, out=keep
    )
    return keep[:, :blocks]


@pytest.mark.parametrize("rows,max_len", [(48, 40000), (32, 9000), (8, 700)])
@torch.inference_mode()
def test_block_masks_match_torch_selection(rows: int, max_len: int) -> None:
    """Level one: fused block keys + ragged top-k keep exactly the blocks the
    padded/pooled torch path keeps (newest block always, nothing unreachable)."""
    logits, lens = _case(rows, max_len, seed=rows * 31 + max_len)
    expected = _reference_position_mask(logits, lens)
    keep = _fused_block_mask(logits, lens)
    positions = keep.bool().repeat_interleave(BLOCK, dim=1)[:, : logits.shape[1]]
    # ties at the 2048th block are the only legitimate difference; random fp32
    # scores have none, so demand equality
    assert torch.equal(positions, expected)
    assert not keep[0].any(), "an empty row keeps no block"


@pytest.mark.parametrize("rows,max_len", [(48, 40000), (16, 3000)])
@torch.inference_mode()
def test_masked_topk_matches_masked_fill(rows: int, max_len: int) -> None:
    """Level two: the masked ragged top-k on the raw scores selects what the
    torch path selects after masking the scores to -inf, including the
    -1 padding of rows with fewer candidates than the top-k."""
    logits, lens = _case(rows, max_len, seed=rows * 17 + max_len)
    keep = _fused_block_mask(logits, lens)
    positions = keep.bool().repeat_interleave(BLOCK, dim=1)[:, : logits.shape[1]]
    offsets = torch.zeros(rows, dtype=torch.int32, device="cuda")

    fused = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    topk_transform_ragged_v2(
        logits.clone(),
        lens,
        out_offsets=offsets,
        out_indices=fused,
        block_mask=keep,
        block_size=BLOCK,
    )

    masked = logits.masked_fill(~positions, -torch.inf)
    expected = torch.empty_like(fused)
    topk_transform_ragged_v2(masked, lens, out_offsets=offsets, out_indices=expected)
    expected = mask_topk_scores(masked, expected, offsets)

    for r in range(rows):
        got = fused[r][fused[r] >= 0]
        want = expected[r][expected[r] >= 0]
        assert got.numel() == want.numel(), (r, got.numel(), want.numel())
        got_v = logits[r, got.long()].sort().values
        want_v = logits[r, want.long()].sort().values
        assert torch.equal(got_v, want_v), r
        assert positions[r, got.long()].all(), r


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
