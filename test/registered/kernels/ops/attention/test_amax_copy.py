"""Correctness tests for the DeepSeek-V4.1 JIT block-max ("amax") copy.

``amax8_varlen`` writes, per row, the maximum of each block of 8 consecutive
fp32 scores for the first ``ceil(seq_lens[b] / 8)`` blocks, the last of them
``+inf`` (the newest block is always selected), and leaves everything else
untouched. It is the level-one key computation of the two-level sparse indexer:
the keys feed a block top-k, which is why rows with at most ``topk`` blocks may
be skipped (every block of such a row is selected anyway).
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import amax8_varlen
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK = 8
SENTINEL = -12345.0
CONFIGS = [
    # (batch, seq): a single partial block, exact blocks, one CTA, many CTAs
    (1, 1),
    (1, 8),
    (1, 9),
    (3, 1000),
    (4, 16389),
    (8, 131072),
    (2, 1048576),
]


def _keys_ref(scores: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Per row the block maxima of the first ceil(len / 8) blocks, the last +inf."""
    batch, width = scores.shape
    nblocks = (width + BLOCK - 1) // BLOCK
    padded = torch.nn.functional.pad(
        scores, (0, nblocks * BLOCK - width), value=-torch.inf
    )
    keys = padded.view(batch, nblocks, BLOCK).amax(-1)
    last = (lens.long() + BLOCK - 1) // BLOCK - 1
    keys[torch.arange(batch, device=scores.device), last] = torch.inf
    return keys


def _check(
    out: torch.Tensor, scores: torch.Tensor, lens: torch.Tensor, skip=None
) -> None:
    ref = _keys_ref(scores, lens)
    for b in range(scores.shape[0]):
        n = (int(lens[b]) + BLOCK - 1) // BLOCK
        if skip is not None and skip[b]:
            assert torch.all(out[b] == SENTINEL), f"row {b} was skipped but written"
            continue
        assert torch.equal(out[b, :n], ref[b, :n]), f"row {b} keys differ"
        assert torch.all(out[b, n:] == SENTINEL), f"row {b} written past its keys"


def _inputs(batch: int, seq: int, ragged: bool, stride_pad: int = 0):
    torch.manual_seed(batch * 977 + seq)
    lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    if ragged:
        lens = torch.randint(1, seq + 1, (batch,), dtype=torch.int32, device="cuda")
        lens[0] = seq
    stride = (seq + BLOCK - 1) // BLOCK * BLOCK + stride_pad
    storage = torch.randn(batch, stride, device="cuda") * 10
    scores = storage[:, :seq]
    return scores, lens


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("batch,seq", CONFIGS)
def test_matches_block_max(batch: int, seq: int, ragged: bool):
    scores, lens = _inputs(batch, seq, ragged)
    out = torch.full((batch, (seq + BLOCK - 1) // BLOCK), SENTINEL, device="cuda")
    assert amax8_varlen(scores, lens, out=out) is out
    _check(out, scores, lens)


def test_strided_views():
    """Score rows padded past the row width and a key buffer wider than needed."""
    scores, lens = _inputs(4, 16389, ragged=True, stride_pad=248)
    nblocks = (16389 + BLOCK - 1) // BLOCK
    out = torch.full((4, nblocks + 5), SENTINEL, device="cuda")
    amax8_varlen(scores, lens, out=out[:, :nblocks])
    _check(out, scores, lens)


def test_allocates_output():
    scores, lens = _inputs(3, 20000, ragged=True)
    out = amax8_varlen(scores, lens)
    assert out.shape == (3, (20000 + BLOCK - 1) // BLOCK) and out.dtype == torch.float32
    ref = _keys_ref(scores, lens)
    for b in range(3):
        n = (int(lens[b]) + BLOCK - 1) // BLOCK
        assert torch.equal(out[b, :n], ref[b, :n])


def test_max_seqlen_sizes_output():
    """A batch whose rows are all shorter than the padded width only needs
    ceil(max_seqlen / 8) keys per row."""
    scores, _ = _inputs(4, 65536, ragged=False)
    lens = torch.tensor([40000, 1, 16384, 39999], dtype=torch.int32, device="cuda")
    out = amax8_varlen(scores, lens, max_seqlen=40000)
    assert out.shape == (4, 5000)
    ref = _keys_ref(scores, lens)
    for b in range(4):
        n = (int(lens[b]) + BLOCK - 1) // BLOCK
        assert torch.equal(out[b, :n], ref[b, :n])


def test_topk_skips_rows_that_fit():
    scores, _ = _inputs(4, 40000, ragged=False)
    lens = torch.tensor([40000, 16384, 16385, 100], dtype=torch.int32, device="cuda")
    out = torch.full((4, 5000), SENTINEL, device="cuda")
    amax8_varlen(scores, lens, 2048, out=out)
    _check(out, scores, lens, skip=[False, True, False, True])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
