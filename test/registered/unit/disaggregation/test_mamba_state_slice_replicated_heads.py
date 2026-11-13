"""CPU tests for GDN conv-state PD slicing when attention TP exceeds key heads.

GDN conv_state is ``cat([q | k | v])``. Value heads divide every TP size, but
key heads (q and k rows) may not: Qwen3-Next / Qwen3.5 have 16 key heads, so
TP 32 or 64 replicates them (rank r holds key head ``r // (tp // 16)``, the
layout ``Mamba2StateShape.create`` sizes for). The slice used to assume every
sub-block is ``full_dim // tp`` wide and fell back to one contiguous slice,
which delivered wrong channels.

Each test builds every rank's local conv rows as logical channel ids, applies
the blocks from ``compute_mamba_state_slice_blocks`` for every (prefill rank,
decode rank) pair SGLang targets, and checks the decode rows.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


import itertools
import unittest

import numpy as np

from sglang.srt.disaggregation.utils import compute_mamba_state_slice_blocks
from sglang.test.test_utils import CustomTestCase


def _heads_on_rank(tp, rank, num_heads):
    if num_heads % tp == 0:
        per = num_heads // tp
        return rank * per, per
    assert tp % num_heads == 0
    return rank // (tp // num_heads), 1


def _local_rows(groups, tp, rank):
    """Logical channel ids of a rank's conv rows: per sub-block, its heads."""
    rows, base = [], 0
    for full_dim, heads in groups:
        head_dim = full_dim // heads
        h0, hn = _heads_on_rank(tp, rank, heads)
        rows.extend(range(base + h0 * head_dim, base + (h0 + hn) * head_dim))
        base += full_dim
    return np.asarray(rows)


def _targets(p_tp, d_tp, d_rank):
    if d_tp >= p_tp:
        return [d_rank // (d_tp // p_tp)]
    k = p_tp // d_tp
    return list(range(d_rank * k, (d_rank + 1) * k))


def transfer(groups, p_tp, d_tp, shard_groups_arg):
    for d in range(d_tp):
        want = _local_rows(groups, d_tp, d)
        got = np.full_like(want, -1)
        writes = np.zeros_like(want)
        for p in _targets(p_tp, d_tp, d):
            src = _local_rows(groups, p_tp, p)
            for s0, d0, n in compute_mamba_state_slice_blocks(
                src_dim=len(src),
                dst_dim=len(want),
                src_attn_tp_size=p_tp,
                dst_attn_tp_size=d_tp,
                dst_tp_rank_in_group=d,
                local_tp_rank_in_group=p,
                conv_shard_groups=shard_groups_arg,
            ):
                got[d0 : d0 + n] = src[s0 : s0 + n]
                writes[d0 : d0 + n] += 1
        np.testing.assert_array_equal(got, want, err_msg=f"decode rank {d}")
        np.testing.assert_array_equal(writes, 1, err_msg=f"decode rank {d} writers")


MODELS = {
    # (key_dim, n_k), (key_dim, n_k), (value_dim, n_v); head dims shrunk.
    "qwen3.5-like nk16 nv64": [(16 * 2, 16), (16 * 2, 16), (64 * 2, 64)],
    "nk2 nv8": [(2 * 4, 2), (2 * 4, 2), (8 * 4, 8)],
    "nk4 nv8 dk!=dv": [(4 * 4, 4), (4 * 4, 4), (8 * 8, 8)],
}


def _tps(groups, candidates):
    nv = groups[2][1]
    return [t for t in candidates if nv % t == 0]


class TestGdnConvSliceReplicatedKeyHeads(CustomTestCase):
    def test_tp_above_key_heads(self):
        """Qwen3.5-like: prefill TP16 <-> decode TP32/64 (key heads replicated)."""
        groups = MODELS["qwen3.5-like nk16 nv64"]
        for p_tp, d_tp in ((16, 32), (16, 64), (32, 64), (64, 16), (64, 32), (8, 64)):
            with self.subTest(p_tp=p_tp, d_tp=d_tp):
                transfer(groups, p_tp, d_tp, groups)

    def test_grid(self):
        for label, groups in MODELS.items():
            tps = _tps(groups, (1, 2, 4, 8, 16, 32, 64))
            for p_tp, d_tp in itertools.product(tps, tps):
                if p_tp == d_tp:
                    continue
                with self.subTest(model=label, p_tp=p_tp, d_tp=d_tp):
                    transfer(groups, p_tp, d_tp, groups)


class TestGdnConvSliceLegacyGroups(CustomTestCase):
    """Bare full dims keep the previous behavior when heads divide both TPs."""

    def test_bare_dims_match_pairs_when_divisible(self):
        groups = MODELS["nk4 nv8 dk!=dv"]
        bare = [full for full, _ in groups]
        for p_tp, d_tp in itertools.product((1, 2, 4), (1, 2, 4)):
            if p_tp == d_tp:
                continue
            for d in range(d_tp):
                for p in _targets(p_tp, d_tp, d):
                    kwargs = dict(
                        src_dim=len(_local_rows(groups, p_tp, p)),
                        dst_dim=len(_local_rows(groups, d_tp, d)),
                        src_attn_tp_size=p_tp,
                        dst_attn_tp_size=d_tp,
                        dst_tp_rank_in_group=d,
                        local_tp_rank_in_group=p,
                    )
                    self.assertEqual(
                        compute_mamba_state_slice_blocks(
                            **kwargs, conv_shard_groups=bare
                        ),
                        compute_mamba_state_slice_blocks(
                            **kwargs, conv_shard_groups=groups
                        ),
                    )
            transfer(groups, p_tp, d_tp, bare)


if __name__ == "__main__":
    unittest.main()
