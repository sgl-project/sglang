"""Fused prefill indexer scores: values and top-k selection vs the torch scoring,
the length mask, and `_score_chunks` with the kernel switched on and off."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer_prefill import fused_index_scores
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    RequestScores,
    _score_chunks,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

HEADS = 32
HEAD_DIM = 128
TOPK = 512
MIN_OVERLAP = 0.99
MIN_EXACT = 0.99
MAX_REL_ERR = 1e-2


def _torch_scores(q, k, weights):
    """`DeepseekV41Indexer.scores`: bf16 up to the head reduction."""
    s = torch.einsum("bhd,nd->bhn", q, k)
    return (s.relu() * weights.unsqueeze(-1)).sum(dim=1).float()


def _inputs(rows, lc, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
    k = torch.randn(lc, HEAD_DIM, generator=g, device="cuda").bfloat16()
    weights = torch.randn(rows, HEADS, generator=g, device="cuda").bfloat16()
    # Prefill tail of a prompt: row i sees lc - rows + 1 + i positions.
    lens = torch.arange(lc - rows + 1, lc + 1, device="cuda").clamp_min(1)
    return q, k, weights, lens


def _mask(scores, lens):
    j = torch.arange(scores.shape[1], device=scores.device)
    return scores.masked_fill(j[None, :] >= lens[:, None], -torch.inf)


def _topk_overlap(a, b, k):
    ia, ib = a.topk(k, dim=-1).indices, b.topk(k, dim=-1).indices
    hits = (ia.unsqueeze(-1) == ib.unsqueeze(-2)).any(-1).sum(-1)
    return (hits.float() / k).mean().item()


class _Indexer:
    def scores(self, q, k, weights):
        return _torch_scores(q, k, weights)


def _chunk_scores(fused, q, k, weights, lens, budget_bytes=None):
    rows = q.shape[0]
    lc = k.shape[0]
    request = RequestScores(
        lc=lc,
        k=min(TOPK, lc),
        columns=torch.arange(lc, device=q.device),
        tok=torch.arange(rows, device=q.device),
        lens=lens,
        slots=torch.arange(lc, device=q.device),
    )
    overrides = [
        envs.SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES.override(fused),
        envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(False),
    ]
    if budget_bytes is not None:
        # Force several row chunks so the fused and torch paths both iterate.
        from unittest.mock import patch

        overrides.append(
            patch.object(scoring, "_TORCH_SCORE_BUDGET_BYTES", budget_bytes)
        )
    from contextlib import ExitStack

    with ExitStack() as stack:
        for ctx in overrides:
            stack.enter_context(ctx)
        return list(
            _score_chunks(
                indexer=_Indexer(), q=q, weights=weights, index_k=k, request=request
            )
        )


def _cat_scores(chunks):
    return torch.cat([chunk.scores for chunk in chunks], dim=0)


@pytest.mark.parametrize("rows,lc", [(1, 700), (77, 1000), (300, 4099), (513, 20011)])
def test_fused_index_scores_random_inputs_match_torch_scoring(rows, lc):
    q, k, weights, lens = _inputs(rows, lc, seed=rows)
    out = fused_index_scores(q, k, weights, lens)
    ref = _mask(_torch_scores(q, k, weights), lens)

    assert out.dtype == torch.float32 and out.shape == (rows, lc)
    assert torch.equal(out.isneginf(), ref.isneginf())
    visible = ref.isfinite()
    # Same bf16 rounding points as torch: values differ only by head summation
    # order, i.e. at most one bf16 step on a few entries.
    exact = (out == ref)[visible].float().mean().item()
    assert exact >= MIN_EXACT, f"bit-equal fraction {exact}"
    rel_err = (out - ref)[visible].abs().max() / ref[visible].abs().max()
    assert rel_err.item() <= MAX_REL_ERR
    overlap = _topk_overlap(out, ref, min(TOPK, int(lens.min())))
    assert overlap >= MIN_OVERLAP, f"top-{TOPK} overlap {overlap}"


@pytest.mark.parametrize("budget_bytes", [None, 1 << 20])
def test_score_chunks_fused_selects_like_torch_scoring(budget_bytes):
    rows, lc = 600, 3001
    q, k, weights, lens = _inputs(rows, lc, seed=2)
    fused_chunks = _chunk_scores(True, q, k, weights, lens, budget_bytes)
    torch_chunks = _chunk_scores(False, q, k, weights, lens, budget_bytes)
    if budget_bytes is not None:
        # lc * 16 bytes per fused row; a 1 MiB budget yields several chunks.
        assert len(fused_chunks) > 1
    fused = _cat_scores(fused_chunks)
    torch_ref = _cat_scores(torch_chunks)

    assert fused.shape == torch_ref.shape == (rows, lc)
    assert torch.equal(fused.isneginf(), torch_ref.isneginf())
    overlap = _topk_overlap(fused, torch_ref, min(TOPK, int(lens.min())))
    assert overlap >= MIN_OVERLAP, f"{budget_bytes=}: overlap {overlap}"


def test_score_chunks_oracle_flag_skips_fused_kernel(monkeypatch):
    """The torch prefill indexer flag is the oracle for the dense fp4 kernel: it
    must score with torch even when the fused kernel is enabled."""

    def _fail(*args, **kwargs):
        raise AssertionError("fused kernel ran under the torch oracle flag")

    monkeypatch.setattr(scoring, "fused_index_scores", _fail)
    rows, lc = 64, 1001
    q, k, weights, lens = _inputs(rows, lc, seed=3)
    request = RequestScores(
        lc=lc,
        k=min(TOPK, lc),
        columns=torch.arange(lc, device=q.device),
        tok=torch.arange(rows, device=q.device),
        lens=lens,
        slots=torch.arange(lc, device=q.device),
    )
    with (
        envs.SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES.override(True),
        envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(True),
    ):
        chunks = list(
            _score_chunks(
                indexer=_Indexer(), q=q, weights=weights, index_k=k, request=request
            )
        )
    scores = _cat_scores(chunks)
    assert scores.shape == (rows, lc)
    assert bool(scores.isfinite().any())
