"""Numeric validation of the two-stage DCP target-verify decomposition.

Under DCP the aiter backend drives aiter's ``mla_decode_fwd`` over each
rank's round-robin KV shard: it tiles the query heads, so it serves any
gathered head count (Kimi-K3 has 96 at tp8 dcp8). Its own causal mask is
``seq_offset < (seq_len - num_tokens_per_seq) + query_pos + 1`` where
``seq_offset`` is the LOCAL shard index, which is wrong under round-robin
sharding for ``q_len > 1``. Target-verify is therefore split in two:

  stage A  mla_decode_fwd over the committed shard, with the ``bs x q_len`` window
           flattened to ``bs*q_len`` single-token rows so ``num_tokens_per_seq
           == 1`` and the mask degenerates to "attend the whole local shard"
           (every committed token precedes every window token, so no masking is
           needed there)
  stage B  dense causal attention over the in-hand ``q_len`` window, folded into
           rank 0 only so the cross-rank merge counts it exactly once
  merge    base-2 LSE merge across ranks, i.e. what ``cp_lse_ag_out_rs_mla``
           does in ``forward_mla``

This test runs the REAL aiter kernel and the REAL reduce/combine helpers against
an fp32 full-attention reference. What it guards, none of which any other test
covers:

* the ``ALL_DECODE = num_tokens_per_seq == 1`` mask degeneration in aiter. An
  aiter bump that changes it makes stage A attend the wrong keys and produces
  *silently* wrong output -- no crash, just a slow drift in speculative accept
  length that is easy to misread as draft-quality noise.
* the LSE log base. ``dense_causal_mla_attn_base2`` rebases its natural-log
  ``logsumexp`` with ``_LOG2E`` because the aiter reduce and sglang's
  ``correct_attn_out`` both work in base-2. Dropping that factor, or swapping
  ``exp2`` for ``exp`` in ``dcp_lse_combine_base2``, silently reweights the merge.
* the empty-shard path (``prefix_len < dcp_size``, so some ranks own nothing and
  must contribute ``lse = -inf`` rather than poisoning the merge with NaN).
* the gathered head counts K3 actually runs (96 at tp8 dcp8), which the kernel
  serves by tiling the query heads; 128 is kept alongside as a control.

Error must also not grow with the shard count: a sharding or merge fault shows
up as error rising with W, which is what ``test_error_does_not_grow_with_w``
pins down.
"""

import itertools
import math
import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")

KV_LORA_RANK = 512
QK_ROPE = 64
D = KV_LORA_RANK + QK_ROPE
PAGE = 32
DTYPE = torch.bfloat16
DEV = "cuda"

# Tolerance is tiered by prefix length. A single flat bound does not work: the
# bf16-vs-fp32 baseline error falls ~10x across the grid (2.4e-3 at prefix 7
# down to 2.6e-4 at prefix 1000, as the softmax mass spreads over more keys),
# so a bound loose enough for the short-prefix tiers is blind at the long ones.
# Each tier is ~2.5x its measured worst case over the whole (W, q_len, H) grid.
#
# Calibrated against three injected faults -- LSE base left un-rebased, stage B
# dropped entirely, empty-shard `-inf` replaced by 0 -- with these results for
# the hardest configuration in each tier:
#
#   prefix     baseline    tol      LSE-base fault   stage-B fault
#   0          2.0e-3      5e-3     (not observable) 1.20
#   1          2.1e-3      5e-3     8.0e-2           0.97
#   7          2.4e-3      6e-3     3.8e-2           0.21
#   100        6.0e-4      1.5e-3   3.6e-3           1.6e-2
#   1000       2.6e-4      1e-3     4.3e-4  <-- see below
#
# Two deliberate blind spots, both inherent rather than sloppy:
#  * prefix 0: no rank owns committed KV, so the cross-stage merge is degenerate
#    and the LSE base cannot affect the result there by construction.
#  * prefix 1000: the fault signal (4.3e-4) sits only 1.65x above the baseline
#    (2.6e-4), too close to separate without making the tier fragile to
#    legitimate kernel rounding changes. This tier therefore guards shape /
#    NaN / multi-page-shard behaviour, not numerics; prefix 1, 7 and 100 carry
#    the numeric guard with 16-24x margin.
MAX_ABS_TOL_BY_PREFIX = {0: 5e-3, 1: 5e-3, 7: 6e-3, 100: 1.5e-3, 1000: 1e-3}

PREFIX_LENS = [0, 1, 7, 100, 1000]
WORLD_SIZES = [1, 2, 4, 8]
Q_LENS = [2, 8]
HEAD_COUNTS = [96, 128]


def _aiter_mla_available() -> bool:
    if not is_hip() or not torch.cuda.is_available():
        return False
    try:
        from aiter.ops.triton.attention.mla import mla_decode_fwd  # noqa: F401
    except Exception:
        return False
    return True


def _reference(q, k_prefix, k_window, bs, q_len, scaling):
    """fp32 full attention: query token i sees all of prefix + window[: i + 1]."""
    qf = q.view(bs, q_len, -1, D).float()
    outs = []
    for b in range(bs):
        keys = torch.cat([k_prefix[b].float(), k_window[b].float()], dim=0)
        n_prefix = k_prefix[b].shape[0]
        scores = torch.einsum("ihd,jd->hij", qf[b], keys) * scaling
        j = torch.arange(keys.shape[0], device=DEV)
        i = torch.arange(q_len, device=DEV)
        allowed = j[None, :] <= (n_prefix + i)[:, None]
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hij,jd->ihd", p, keys[:, :KV_LORA_RANK]))
    return torch.stack(outs).reshape(bs * q_len, -1, KV_LORA_RANK)


def _stage_a(q, shard, shard_lens, bs, q_len, num_heads, scaling):
    """mla_decode_fwd over one rank's committed shard, flattened to bs*q_len rows."""
    from aiter.ops.triton.attention.mla import mla_decode_fwd

    from sglang.kernels.ops.attention.dcp_kernels import dcp_reduce_kv_segments

    n_rows = bs * q_len
    max_local = int(shard_lens.max().item())
    if max_local == 0:
        # mla_decode_fwd cannot be launched with a zero-length shard (it would
        # get a zero-page block table), so stand in for it with the identity
        # element of the merge. NOTE this stands in for the KERNEL only -- it
        # deliberately does not assert anything about dcp_reduce_kv_segments, whose
        # own empty-shard contract is covered by
        # TestDcpReduceKvSegmentsEmptyShard below.
        return (
            torch.zeros(n_rows, num_heads, KV_LORA_RANK, dtype=DTYPE, device=DEV),
            torch.full((n_rows, num_heads), float("-inf"), device=DEV),
        )

    max_pages = (max_local + PAGE - 1) // PAGE
    # Per-request paged shard buffer; the tail page is zero-padded because the kernel
    # reads the whole last page and masks it by seqused_k.
    paged = torch.zeros(bs * max_pages, PAGE, 1, D, dtype=DTYPE, device=DEV)
    block_tables = torch.zeros(bs, max_pages, dtype=torch.int32, device=DEV)
    for b in range(bs):
        n = int(shard_lens[b].item())
        if n:
            paged.view(bs, max_pages * PAGE, 1, D)[b, :n] = shard[b][:n].unsqueeze(1)
        block_tables[b] = torch.arange(max_pages, device=DEV) + b * max_pages

    seqused = shard_lens.repeat_interleave(q_len).to(torch.int32)
    out = torch.empty(n_rows, num_heads, KV_LORA_RANK, dtype=DTYPE, device=DEV)
    segm_o, segm_m, segm_e = mla_decode_fwd(
        q.view(n_rows, num_heads, D),
        paged,
        out,
        torch.arange(n_rows + 1, dtype=torch.int32, device=DEV),
        seqused,
        max_local,
        block_tables.repeat_interleave(q_len, dim=0),
        scaling,
        KV_LORA_RANK,
        QK_ROPE,
        True,
        None,
        None,
        skip_reduce=True,
    )
    return dcp_reduce_kv_segments(segm_o, segm_m, segm_e, seqused, PAGE, DTYPE)


def _merge_ranks(outs, lses):
    """Base-2 LSE merge across ranks == cp_lse_ag_out_rs_mla / correct_attn_out."""
    lse_all = torch.stack(lses)
    m = lse_all.max(dim=0).values
    w = torch.nan_to_num(torch.exp2(lse_all - m), nan=0.0, posinf=0.0, neginf=0.0)
    num = sum(o.float() * w[r].unsqueeze(-1) for r, o in enumerate(outs))
    return num / w.sum(dim=0).clamp_min(1e-38).unsqueeze(-1)


def _run_config(bs, q_len, num_heads, prefix_len, world_size):
    """Return (max_abs_err, has_nan) for one simulated DCP group."""
    from sglang.kernels.ops.attention.dcp_kernels import (
        dcp_lse_combine_base2,
        dense_causal_mla_attn_base2,
    )

    torch.manual_seed(0)
    scaling = 1.0 / math.sqrt(D)
    q = torch.randn(bs * q_len, num_heads, D, dtype=DTYPE, device=DEV) * 0.3
    k_prefix = torch.randn(bs, prefix_len, D, dtype=DTYPE, device=DEV) * 0.3
    k_window = torch.randn(bs, q_len, D, dtype=DTYPE, device=DEV) * 0.3

    outs, lses = [], []
    for rank in range(world_size):
        # Round-robin owner rule: rank r holds committed positions p % W == r.
        idx = (
            torch.arange(rank, prefix_len, world_size, device=DEV)
            if prefix_len > rank
            else torch.empty(0, dtype=torch.long, device=DEV)
        )
        shard = k_prefix[:, idx, :]
        shard_lens = torch.full((bs,), idx.numel(), dtype=torch.int32, device=DEV)
        out_r, lse_r = _stage_a(q, shard, shard_lens, bs, q_len, num_heads, scaling)
        if rank == 0:
            out_b, lse_b = dense_causal_mla_attn_base2(
                q,
                k_window.reshape(bs * q_len, 1, D),
                scaling,
                bs,
                q_len,
                KV_LORA_RANK,
            )
            out_r, lse_r = dcp_lse_combine_base2(out_r, lse_r, out_b, lse_b, DTYPE)
        outs.append(out_r)
        lses.append(lse_r)

    got = _merge_ranks(outs, lses)
    ref = _reference(q, k_prefix, k_window, bs, q_len, scaling)
    return (got - ref).abs().max().item(), bool(torch.isnan(got).any())


@unittest.skipUnless(
    _aiter_mla_available(), "requires ROCm + aiter with the MLA decode kernel"
)
class TestDcpVerifyDecomposition(CustomTestCase):
    def test_matches_full_attention_reference(self):
        """Two-stage verify == fp32 full attention, across the DCP grid."""
        for prefix_len, world_size, q_len, num_heads in itertools.product(
            PREFIX_LENS, WORLD_SIZES, Q_LENS, HEAD_COUNTS
        ):
            tol = MAX_ABS_TOL_BY_PREFIX[prefix_len]
            with self.subTest(
                prefix_len=prefix_len, W=world_size, q_len=q_len, H=num_heads
            ):
                err, has_nan = _run_config(4, q_len, num_heads, prefix_len, world_size)
                self.assertFalse(has_nan, "NaN in the merged attention output")
                self.assertTrue(math.isfinite(err), f"non-finite error: {err}")
                self.assertLess(err, tol, f"max_abs {err:.6f} exceeds {tol:.1e}")

    def test_error_does_not_grow_with_w(self):
        """Sharding must not degrade accuracy as the DCP group grows.

        A correct decomposition is W-independent: each rank attends a disjoint
        key subset and the LSE merge is exact. A sharding or merge fault instead
        degrades progressively with W, so the trend is a sharper signal than any
        single absolute error.
        """
        for prefix_len, q_len in itertools.product([7, 100, 1000], Q_LENS):
            with self.subTest(prefix_len=prefix_len, q_len=q_len):
                base, _ = _run_config(4, q_len, 96, prefix_len, 1)
                for world_size in (2, 4, 8):
                    err, _ = _run_config(4, q_len, 96, prefix_len, world_size)
                    # Absolute floor absorbs bf16 jitter when `base` is tiny.
                    self.assertLessEqual(
                        err,
                        max(base * 2.0, 5e-4),
                        f"W={world_size} error {err:.6f} exceeds the W=1 "
                        f"baseline {base:.6f}; sharding is degrading accuracy",
                    )


@unittest.skipUnless(
    _aiter_mla_available(), "requires ROCm + aiter with the MLA decode kernel"
)
class TestDcpReduceKvSegmentsEmptyShard(CustomTestCase):
    """dcp_reduce_kv_segments's contract for a rank that owns no committed KV.

    A rank owns nothing for a request whose prefix is shorter than its rank
    index, and the planner clamps local_kv_lens to min 0, so a zero-length row
    is a normal input. The reduce must emit the identity element of the LSE
    merge -- out 0, lse -inf -- which dcp_lse_combine_base2 and cp_lse_ag_out_rs_mla
    both weight to zero.

    Regression: the reduce used to compute act_num_segments as
    ``cdiv(seq_len, cdiv(seq_len, N*T) * T)``, which divides by zero at
    seq_len 0. The resulting garbage segment mask produced lse NaN instead of
    -inf, and the NaN survived the merge -- nan_to_num zeroes the WEIGHT, but
    NaN * 0 is still NaN, so one empty shard poisoned the merged output of an
    otherwise healthy batch.

    The grid in TestDcpVerifyDecomposition cannot reach this: every request
    there shares one prefix_len, so a rank's shard lengths are uniform and the
    all-empty case short-circuits before the kernel runs. The mixed case below
    is the one that occurs in practice -- a short prompt batched with a long one.
    """

    def _reduce(self, shard_lens):
        from sglang.kernels.ops.attention.dcp_kernels import dcp_reduce_kv_segments

        torch.manual_seed(0)
        n_rows, heads, segments = len(shard_lens), 2, 4
        # normal_(), not zeros: aiter hands the reduce an empty() buffer whose
        # inactive segments hold garbage, and the masking is what this guards.
        shape = (n_rows, heads, segments)
        segm_o = torch.empty(*shape, KV_LORA_RANK, device=DEV).normal_(0, 1.0)
        segm_m = torch.empty(*shape, device=DEV).normal_(0, 1.0)
        segm_e = torch.empty(*shape, device=DEV).normal_(0, 1.0).abs()
        lens = torch.tensor(shard_lens, dtype=torch.int32, device=DEV)
        return dcp_reduce_kv_segments(segm_o, segm_m, segm_e, lens, PAGE, DTYPE)

    def test_mixed_batch_empty_rows_are_the_merge_identity(self):
        shard_lens = [64, 0, 8, 0]
        out, lse = self._reduce(shard_lens)
        for row, n in enumerate(shard_lens):
            with self.subTest(row=row, shard_len=n):
                self.assertTrue(
                    torch.isfinite(out[row].float()).all(), "non-finite output"
                )
                if n == 0:
                    self.assertTrue((out[row] == 0).all(), "empty shard must emit 0")
                    self.assertTrue(
                        torch.isneginf(lse[row]).all(), "empty shard must emit -inf"
                    )
                else:
                    self.assertTrue(
                        torch.isfinite(lse[row]).all(),
                        "a non-empty row must not be disturbed by an empty neighbour",
                    )

    def test_empty_shard_does_not_poison_the_merge(self):
        """The end that actually matters: a NaN here corrupts real output."""
        from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_base2

        out_a, lse_a = self._reduce([16, 0])
        out_b = torch.ones(2, 2, KV_LORA_RANK, device=DEV)
        lse_b = torch.zeros(2, 2, device=DEV)
        merged, _ = dcp_lse_combine_base2(out_a, lse_a, out_b, lse_b, torch.float32)
        self.assertTrue(torch.isfinite(merged).all(), "empty shard poisoned the merge")
        # Row 1's stage A saw no keys, so the merge must return stage B untouched.
        torch.testing.assert_close(merged[1], out_b[1].float())

    def test_merge_of_two_empty_partials(self):
        """Both sides empty: the merge weights sum to zero.

        Neither of the two comparable kernels survives this -- merge_state_kernel
        computes `-inf - (-inf)` and `_dcp_lse_combine_kernel` divides `0 / 0`,
        both NaN. Reachable at prefix 0, where no rank owns committed KV and a
        non-rank-0 partial has no window to fold in either.
        """
        from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_base2

        out_a, lse_a = self._reduce([0, 0])
        merged, lse = dcp_lse_combine_base2(
            out_a, lse_a, out_a.clone(), lse_a.clone(), torch.float32
        )
        self.assertTrue(torch.isfinite(merged).all(), "all-empty merge produced NaN")
        self.assertTrue((merged == 0).all(), "all-empty merge must emit 0")
        self.assertTrue(torch.isneginf(lse).all(), "all-empty merge must emit -inf")


if __name__ == "__main__":
    unittest.main()
