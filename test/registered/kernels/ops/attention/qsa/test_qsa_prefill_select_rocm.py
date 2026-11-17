"""gfx95 QSA prefill block selection through ``QSAIndexer.select_prefill_tokens``.

On gfx95 the blocks come from topk_v2, which selects exactly where ``fast_topk``
keeps at most 4096 candidates of a row's threshold bin and drops the rest, and in
batch-invariant mode gives ties to the smallest block indices.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa.mqa import (
    HAS_TILELANG,
    tilelang_qsa_mqa_prefill_relative,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = [
    pytest.mark.skipif(not is_gfx95_supported(), reason="gfx95-only selection path"),
    pytest.mark.skipif(not HAS_TILELANG, reason="tilelang unavailable"),
]

HEAD_DIM = 128
NUM_Q_HEADS = 4
RATIO = 4
BLOCK_TOPK = 512
TOKEN_TOPK = 2048
ROWS_PER_REQUEST = 32
SECOND_REQUEST_KEYS = 1000


def _select(q, k, key_counts):
    """select_prefill_tokens on q and k (bf16, on the GPU) over requests packed back to
    back, one per entry of key_counts (its compressed keys): the last ROWS_PER_REQUEST
    tokens of each request are query rows, and a row sees the blocks completed before
    it. Returns the selected tokens and each row's start and end in the keys."""
    starts, ends, positions, seq_lens, base = [], [], [], [], 0
    for count in key_counts:
        seq_len = RATIO * count + 1
        row_positions = torch.arange(seq_len - ROWS_PER_REQUEST, seq_len)
        starts.append(torch.full((ROWS_PER_REQUEST,), base))
        ends.append(base + (row_positions + 1) // RATIO)
        positions.append(row_positions)
        seq_lens.append(torch.full((ROWS_PER_REQUEST,), seq_len))
        base += count
    device = q.device
    starts = torch.cat(starts).int().to(device)
    ends = torch.cat(ends).int().to(device)
    indexer = SimpleNamespace(
        token_topk=TOKEN_TOPK, compress_ratio=RATIO, block_topk=BLOCK_TOPK
    )
    selected = QSAIndexer.select_prefill_tokens(
        indexer,
        q,
        k,
        starts,
        ends,
        torch.cat(positions).to(device),
        torch.cat(seq_lens).int().to(device),
        max_row_keys=max(key_counts),
    )
    return selected, starts, ends


def _blocks(selected_row, length):
    tokens = selected_row[selected_row >= 0].long()
    blocks = torch.unique(tokens // RATIO)
    return blocks[blocks < length]  # the row's incomplete group is not a block


# 8190 is not a multiple of 4: topk_v2 rejects such a logits width unless it is rounded up.
@pytest.mark.parametrize("num_keys", [8192, 8190])
def test_select_prefill_tokens_exact_on_narrow_band_scores(num_keys):
    """Scores packed into one fp16 coarse bin (all about 22.6 +- 1): a row of the first
    request has about 8,190 members of its threshold bin, twice the 4096 that fast_topk
    stages, so it dropped members of the true top-k. Each row's selected blocks,
    relative to its request, must hold exactly its top-512 scores. A second, shorter
    request packed behind the first starts at a nonzero column of the keys."""
    gen = torch.Generator().manual_seed(7)
    # With q all ones, a head's score of key j is relu(sum_d k[j, d]) = 64 +- 0.6.
    key_counts = (num_keys, SECOND_REQUEST_KEYS)
    k = 0.5 + 0.05 * torch.randn(sum(key_counts), 1, HEAD_DIM, generator=gen)
    q = torch.ones(len(key_counts) * ROWS_PER_REQUEST, NUM_Q_HEADS, HEAD_DIM)
    q, k = q.bfloat16().cuda(), k.bfloat16().cuda()

    selected, starts, ends = _select(q, k, key_counts)

    scores = tilelang_qsa_mqa_prefill_relative(q, k, starts, ends, num_keys)
    for row in range(q.shape[0]):
        length = int(ends[row] - starts[row])
        blocks = _blocks(selected[row], length)
        assert blocks.numel() == BLOCK_TOPK, row
        expected = torch.topk(scores[row, :length], BLOCK_TOPK).values.sort().values
        assert torch.equal(scores[row, blocks].sort().values, expected), row


def test_select_prefill_tokens_ties_go_to_the_smallest_blocks_when_batch_invariant(
    monkeypatch,
):
    """Every key equal, so all of a row's scores tie: in batch-invariant mode each row
    must select blocks 0..511, the smallest indices."""
    monkeypatch.setattr(qsa_indexer, "is_batch_invariant_mode_enabled", lambda: True)
    gen = torch.Generator().manual_seed(7)
    num_keys = 4096
    k = torch.full((num_keys, 1, HEAD_DIM), 0.05)
    q = torch.randn(ROWS_PER_REQUEST, NUM_Q_HEADS, HEAD_DIM, generator=gen)
    q, k = q.bfloat16().cuda(), k.bfloat16().cuda()

    selected, starts, ends = _select(q, k, (num_keys,))

    scores = tilelang_qsa_mqa_prefill_relative(q, k, starts, ends, num_keys)
    for row in range(q.shape[0]):
        length = int(ends[row] - starts[row])
        assert torch.all(scores[row, :length] == scores[row, 0]), row
        blocks = _blocks(selected[row], length)
        assert torch.equal(blocks.cpu(), torch.arange(BLOCK_TOPK)), row


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
