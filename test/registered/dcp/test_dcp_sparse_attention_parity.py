"""P3a's exit criterion: sharded sparse attention equals unsharded, in pure torch.

[Test Category] Correctness
[Test Target] The DCP sparse-attention composition -- remap, shard, merge

Proves the whole P3a scheme end to end without an attention operator, without
an LSE from silicon and without an NPU:

    dense    top-k over global positions -> attention -> output
    sharded  top-k -> remap to rank-local -> attention on each rank's KV shard
             -> LSE-weighted combine -> output

These must agree. If they do, the sharding is right and any later failure is in
the operator or the LSE base -- which is exactly the boundary the plan draws
between P3a (this workstream) and P3b (the sgl-kernel-npu SFA port). That split
only pays off if this test exists and passes first, so this is the acceptance
test P3b will be measured against, not just a unit test.

Everything here is deliberately naive. The reference attention is written out
in three lines rather than reusing a fast path, because the point is to
establish what the answer IS, independently of any kernel that computes it.

WHY THE LSE IS BASE-E HERE. The merge weights partial outputs by exp(lse_r -
max), so the reference and the combine have to agree on what log the LSE is in.
This file uses natural log throughout, matching vLLM-Ascend, whose SFA-CP path
forms it as ``softmax_max + torch.log(softmax_sum)``
(``attention/context_parallel/sfa_cp.py:1250``) -- the only working reference
for DCP composed with a sparse indexer. See test_the_wrong_log_base_is_not
_silently_fine below: getting this wrong does not raise, it degrades.

Usage:
    python -m pytest test_dcp_sparse_attention_parity.py -v
    python test_dcp_sparse_attention_parity.py
"""

import math
import unittest

import torch

from sglang.srt.layers.dcp.layout import remap_dcp_local_topk_indices
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS = 4
HEAD_DIM = 32
PAD = -1
NEG_INF = float("-inf")


def _attend(q, kv, idx, scale):
    """Attention restricted to the positions named in ``idx``.

    Returns ``(out [H, D], lse [H])`` with a natural-log LSE, the FlashAttention
    convention: ``lse = m + log(sum(exp(s - m)))``.

    Entries of ``idx`` below zero are padding and are skipped -- the sentinel is
    the whole reason the remap can keep a fixed row width. A rank that owns none
    of the selected positions returns a zero output and an LSE of -inf, which is
    the identity element of the combine.
    """
    valid = idx[idx >= 0].to(torch.int64)
    if valid.numel() == 0:
        return (
            torch.zeros(HEADS, HEAD_DIM, dtype=q.dtype),
            torch.full((HEADS,), NEG_INF, dtype=q.dtype),
        )
    k = kv[valid]  # [n, D]
    scores = (q @ k.T) * scale  # [H, n]
    m = scores.max(dim=-1).values
    e = torch.exp(scores - m[:, None])
    denom = e.sum(dim=-1)
    out = (e @ kv[valid]) / denom[:, None]
    return out, m + torch.log(denom)


def _shard(kv, dcp_size, rank):
    """Rank r's physical KV under the owner rule ``pos % c == r``.

    Row ``local`` of this holds global position ``local * c + r`` -- which is
    precisely the inverse the remap computes, so indexing the shard with a
    remapped index has to land on the same token.
    """
    return kv[rank::dcp_size]


def _combine(partial_out, partial_lse, base_e=True):
    """LSE-weighted merge of per-rank partials: [N,H,D] + [N,H] -> [H,D].

    Deliberately re-implemented rather than imported. The in-tree
    ``_lse_weighted_combine_cpu`` lives in ``kernels/ops/attention/dcp_kernels.py``,
    which imports triton at module scope -- which is why the one existing test
    that uses it is registered as CUDA CI. This file has to run on a plain CPU
    runner, and P3a's exit criterion should not be contingent on triton being
    installed to prove index arithmetic.

    ``test_the_inline_combine_matches_the_in_tree_one`` pins the two together
    wherever the import does work, so this stays a copy rather than a fork.
    """
    lse = torch.where(
        torch.isnan(partial_lse) | torch.isinf(partial_lse),
        torch.full_like(partial_lse, NEG_INF),
        partial_lse,
    )
    lse_max = lse.max(dim=0).values
    lse_max = torch.where(lse_max == NEG_INF, torch.zeros_like(lse_max), lse_max)
    centered = lse - lse_max.unsqueeze(0)
    w = torch.exp(centered) if base_e else torch.pow(2.0, centered)
    w = w / w.sum(dim=0, keepdim=True)
    return (partial_out * w.unsqueeze(-1)).sum(dim=0)


def _remap(topk, dcp_size, rank):
    with get_parallel().override(
        dcp_enabled=dcp_size > 1,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=rank,
    ):
        return remap_dcp_local_topk_indices(topk)


def _sharded_attention(q, kv, topk, dcp_size, scale, *, base_e=True):
    """Run every rank and merge, the way a real decode step would."""
    outs, lses = [], []
    for rank in range(dcp_size):
        local = _remap(topk, dcp_size, rank)[0]
        out, lse = _attend(q, _shard(kv, dcp_size, rank), local, scale)
        outs.append(out)
        lses.append(lse)
    return _combine(torch.stack(outs), torch.stack(lses), base_e=base_e)


def _fixture(seq_len, k, seed):
    """A query, a KV cache, and a top-k row of distinct global positions."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(HEADS, HEAD_DIM, generator=g)
    kv = torch.randn(seq_len, HEAD_DIM, generator=g)
    n = min(k, seq_len)
    picks = torch.randperm(seq_len, generator=g)[:n].to(torch.int32)
    topk = torch.cat([picks, torch.full((k - n,), PAD, dtype=torch.int32)])
    return q, kv, topk.unsqueeze(0)


class TestDcpSparseAttentionParity(CustomTestCase):
    SCALE = 1.0 / math.sqrt(HEAD_DIM)

    def _assert_parity(self, seq_len, k, dcp_size, seed):
        q, kv, topk = _fixture(seq_len, k, seed)
        dense, _ = _attend(q, kv, topk[0], self.SCALE)
        sharded = _sharded_attention(q, kv, topk, dcp_size, self.SCALE)
        torch.testing.assert_close(sharded, dense, rtol=1e-4, atol=1e-5)

    def test_sharded_equals_dense(self):
        """The claim P3a exists to establish."""
        for dcp_size in (2, 3, 4, 8, 16):
            for seq_len, k in ((256, 32), (1024, 64), (97, 16)):
                with self.subTest(dcp_size=dcp_size, seq_len=seq_len, k=k):
                    self._assert_parity(seq_len, k, dcp_size, seed=100 + seq_len)

    def test_it_holds_when_the_top_k_is_the_whole_sequence(self):
        # Dense attention as a special case of sparse: every position selected,
        # so the shards partition the entire KV cache and nothing is dropped.
        for dcp_size in (2, 4, 16):
            with self.subTest(dcp_size=dcp_size):
                self._assert_parity(128, 128, dcp_size, seed=7)

    def test_it_holds_when_ranks_own_nothing(self):
        """seq_len < dcp_size, which happens at the very start of decode.

        Most ranks own no tokens at all and contribute an all-padding row, a
        zero output and an LSE of -inf. The combine has to treat those as
        weightless rather than as zeros to average in -- a bug here shows up as
        an output scaled by (owned ranks / dcp_size), which looks like a
        temperature change rather than a crash.
        """
        for dcp_size in (4, 8, 16):
            for seq_len in (1, 2, 3):
                with self.subTest(dcp_size=dcp_size, seq_len=seq_len):
                    self._assert_parity(seq_len, 8, dcp_size, seed=11)

    def test_every_selected_position_is_attended_exactly_once(self):
        """Structural check behind the numeric one.

        Parity could in principle be reached with a compensating pair of errors
        -- a dropped position and a duplicated one. This asserts the union of
        what the ranks actually read, in global coordinates, is exactly the
        selected set.
        """
        for dcp_size in (2, 4, 16):
            q, kv, topk = _fixture(512, 48, seed=23)
            with self.subTest(dcp_size=dcp_size):
                seen = []
                for rank in range(dcp_size):
                    local = _remap(topk, dcp_size, rank)[0]
                    for i in local[local >= 0].tolist():
                        seen.append(i * dcp_size + rank)
                expected = sorted(p for p in topk[0].tolist() if p >= 0)
                self.assertEqual(sorted(seen), expected)
                self.assertEqual(len(seen), len(set(seen)))

    def test_the_shard_index_lands_on_the_token_the_remap_meant(self):
        # The remap and the physical shard layout are two halves of one
        # convention. Pinning them against each other catches the case where
        # both are self-consistent but disagree by a rank.
        kv = torch.arange(64, dtype=torch.float32).reshape(64, 1)
        for dcp_size in (2, 3, 4, 8):
            topk = torch.arange(64, dtype=torch.int32).unsqueeze(0)
            for rank in range(dcp_size):
                with self.subTest(dcp_size=dcp_size, rank=rank):
                    local = _remap(topk, dcp_size, rank)[0]
                    shard = _shard(kv, dcp_size, rank)
                    for i in local[local >= 0].tolist():
                        self.assertEqual(float(shard[i, 0]), float(i * dcp_size + rank))

    def test_the_wrong_log_base_is_not_silently_fine(self):
        """Guards the finding this file's header records.

        If the operator emits a natural-log LSE and the merge is told it is
        base-2, nothing raises: exp2 is a monotone reweighting, so the output
        stays finite and plausible. That is the failure mode the plan warns
        about -- acceptance falls, accuracy tests still pass. This asserts the
        mismatch is at least *detectable*, so that when P3b lands there is a
        test that fails rather than a metric that drifts.
        """
        q, kv, topk = _fixture(512, 48, seed=31)
        dense, _ = _attend(q, kv, topk[0], self.SCALE)
        right = _sharded_attention(q, kv, topk, 8, self.SCALE, base_e=True)
        wrong = _sharded_attention(q, kv, topk, 8, self.SCALE, base_e=False)

        torch.testing.assert_close(right, dense, rtol=1e-4, atol=1e-5)
        self.assertFalse(
            torch.allclose(wrong, dense, rtol=1e-4, atol=1e-5),
            "a base-2 merge of base-e LSEs matched the reference; either the "
            "fixture is too flat to distinguish them or the combine ignores "
            "is_lse_base_on_e",
        )
        # It stays finite and same-magnitude, which is exactly why it is
        # dangerous -- not a crash, just a differently-weighted answer.
        self.assertTrue(torch.isfinite(wrong).all())

    def test_ascend_is_declared_a_natural_log_backend(self):
        """The host-side half of the test above, and the one that regresses.

        The test above proves the mismatch is detectable. This one proves the
        merge is actually told the right base on this target: the selector is
        an allowlist, so a backend absent from it silently gets base-2. CANN
        defines softmax_sum as sum(exp(qk - max)) and the LSE is reconstructed
        as softmax_max + log(softmax_sum) -- natural log -- which is also what
        the CANN family's own golden reference computes and what vLLM-Ascend's
        DCP path forms before merging.

        Lives here rather than only in
        test/registered/kernels/test_dcp_lse_combine.py, which is CUDA CI, so
        without this the assertion never runs on the target it is about. Same
        skip as the test below and for the same reason: forward_mla imports
        layers.dcp, which imports triton at module scope. That skips on a bare
        CPU runner and *runs* on the NPU box, where triton-ascend installs as
        `triton` -- which is the machine this assertion is about.
        """
        try:
            from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
                is_mla_dcp_lse_base_on_e,
            )
        except ImportError as exc:  # pragma: no cover - depends on the runner
            self.skipTest(f"forward_mla needs triton via layers.dcp: {exc}")

        self.assertTrue(is_mla_dcp_lse_base_on_e("ascend"))
        # Not a blanket opt-in: backends that really do return base-2 must stay
        # out, or this fix trades one silent mis-weighting for another.
        self.assertFalse(is_mla_dcp_lse_base_on_e("flashinfer_mla"))
        self.assertFalse(is_mla_dcp_lse_base_on_e(None))

    def test_the_inline_combine_matches_the_in_tree_one(self):
        """Keeps _combine above honest against the reference the plan names.

        Skipped rather than failed where triton is missing: dcp_kernels.py
        imports it at module scope, and the point of the skip is that the
        parity proof itself does not depend on this.
        """
        try:
            from sglang.kernels.ops.attention.dcp_kernels import (
                _lse_weighted_combine_cpu,
            )
        except ImportError as exc:  # pragma: no cover - depends on the runner
            self.skipTest(f"dcp_kernels needs triton: {exc}")

        g = torch.Generator().manual_seed(53)
        for base_e in (True, False):
            for lses in (
                torch.randn(8, 1, HEADS, generator=g) * 3,
                # an empty shard and a poisoned entry, the cases where the
                # sanitize branch is what is actually being compared
                torch.tensor([[[NEG_INF] * HEADS]] * 4 + [[[1.0] * HEADS]] * 4),
            ):
                outs = torch.randn(8, 1, HEADS, HEAD_DIM, generator=g)
                with self.subTest(base_e=base_e):
                    theirs = _lse_weighted_combine_cpu(
                        outs, lses, is_lse_base_on_e=base_e
                    )
                    mine = _combine(outs[:, 0], lses[:, 0], base_e=base_e)
                    torch.testing.assert_close(mine, theirs[0], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
