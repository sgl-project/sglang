"""Unit tests for ``should_skip_post_experts_all_reduce``.

This pins the contract that decides whether the model's post-experts all-reduce
(the TP/EP reduction applied after ``self.experts(...)`` in the standard MoE
forward) is skipped because a downstream component already reduces.

The case that matters most here is ``pplx``: its ``AllToAll.combine`` already
sums each token's expert outputs back to the source rank (a complete per-rank
result), exactly like the ``flashinfer`` A2A dispatcher. If the model then also
runs the post-experts all-reduce, the result is double-counted -- and under DP
attention with idle ranks (fewer concurrent requests than ``dp_size``) it folds
the idle ranks' fabricated outputs into the real tokens, producing garbage. So
``pplx`` must skip that all-reduce. This test is the regression guard for that.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    should_skip_post_experts_all_reduce,
)
from sglang.test.test_utils import CustomTestCase


@contextmanager
def _backend(name, *, dp_reduce_scatterv=False, flashinfer_fp4_allgather=False):
    """Pin the module-level globals ``should_skip_...`` reads, isolated from any
    real server/parallel state so this runs on CPU CI."""
    with patch.object(
        moe_utils, "get_moe_a2a_backend", return_value=MoeA2ABackend(name)
    ), patch.object(
        moe_utils,
        "should_use_dp_reduce_scatterv",
        return_value=dp_reduce_scatterv,
    ), patch.object(
        moe_utils,
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        return_value=flashinfer_fp4_allgather,
    ):
        yield


class TestShouldSkipPostExpertsAllReduce(CustomTestCase):
    def test_pplx_skips_because_combine_already_reduces(self):
        # The fix: pplx's combine already reduces cross-rank, so the post-experts
        # all-reduce must be skipped on both the TP and EP call sites.
        with _backend("pplx"):
            self.assertTrue(
                should_skip_post_experts_all_reduce(is_tp_path=True),
                "pplx must skip the post-experts TP all-reduce",
            )
            self.assertTrue(
                should_skip_post_experts_all_reduce(is_tp_path=False),
                "pplx must skip the post-experts EP all-reduce",
            )

    def test_flashinfer_skips_like_pplx(self):
        # Sibling case the pplx fix mirrors; guards against the clause being
        # removed while refactoring.
        with _backend("flashinfer"):
            self.assertTrue(should_skip_post_experts_all_reduce(is_tp_path=True))
            self.assertTrue(should_skip_post_experts_all_reduce(is_tp_path=False))

    def test_none_does_not_skip(self):
        # Plain TP MoE genuinely needs the all-reduce; skipping it would corrupt
        # output. Ensure the pplx clause did not widen the skip to every backend.
        with _backend("none"):
            self.assertFalse(should_skip_post_experts_all_reduce(is_tp_path=True))
            self.assertFalse(should_skip_post_experts_all_reduce(is_tp_path=False))

    def test_deepep_does_not_skip_here(self):
        # deepep/mooncake route through ``forward_deepep`` and never reach this
        # helper, so it legitimately returns False for them (they are not in the
        # skip list). Pin that so nobody "fixes" it by adding them here.
        with _backend("deepep"):
            self.assertFalse(should_skip_post_experts_all_reduce(is_tp_path=True))

    def test_fusion_and_reduce_scatter_flags_force_skip_for_any_backend(self):
        # Independent of backend: a downstream fusion / reduce-scatter absorbs the
        # all-reduce, so it must be skipped.
        with _backend("none"):
            self.assertTrue(
                should_skip_post_experts_all_reduce(
                    is_tp_path=True, should_allreduce_fusion=True
                )
            )
            self.assertTrue(
                should_skip_post_experts_all_reduce(
                    is_tp_path=True, use_reduce_scatter=True
                )
            )

    def test_dp_reduce_scatterv_forces_skip(self):
        with _backend("none", dp_reduce_scatterv=True):
            self.assertTrue(should_skip_post_experts_all_reduce(is_tp_path=True))


if __name__ == "__main__":
    unittest.main(verbosity=3)
