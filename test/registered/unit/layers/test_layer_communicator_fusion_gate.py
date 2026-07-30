import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.layers.moe import post_experts_all_reduce
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator():
    """Minimal stand-in for the attributes should_fuse_mlp_allreduce_with_next_layer reads."""
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=ScatterMode.TP_ATTN_FULL),
        is_last_layer=False,
        _context=types.SimpleNamespace(tp_size=4),
    )


class TestPostExpertsAllReduceMerge(unittest.TestCase):
    """The two post-experts reductions collapse into one _TP reduction.

    _MOE_EP and _MOE_TP are orthogonal subgroups of _TP, so with
    moe_dp_size == 1 reducing over each in turn equals one _TP reduction --
    one collective instead of two. With moe_dp_size > 1 they cover only part of
    _TP and merging would sum across DP replicas, which hold different tokens.
    """

    def _calls(self, *, moe_ep_size, moe_tp_size, moe_dp_size=1, skip=False):
        """Which all-reduce helpers post_experts_all_reduce() invokes."""
        called = []

        def record(name):
            return lambda x: called.append(name) or x

        with patch.object(
            moe_utils, "should_skip_post_experts_all_reduce", return_value=skip
        ), patch(
            "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
            side_effect=record("tp"),
        ), patch(
            "sglang.srt.distributed.communication_op.moe_expert_parallel_all_reduce",
            side_effect=record("ep"),
        ), patch(
            "sglang.srt.distributed.communication_op.moe_tensor_model_parallel_all_reduce",
            side_effect=record("moe_tp"),
        ), get_parallel().override(
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=moe_dp_size,
            tp_size=moe_ep_size * moe_tp_size * moe_dp_size,
        ):
            post_experts_all_reduce(torch.zeros(2, 2))
        return called

    def test_hybrid_issues_one_tp_reduction(self):
        self.assertEqual(self._calls(moe_ep_size=2, moe_tp_size=2), ["tp"])

    def test_moe_dp_keeps_the_two_step_form(self):
        self.assertEqual(
            self._calls(moe_ep_size=2, moe_tp_size=2, moe_dp_size=2), ["ep", "moe_tp"]
        )

    def test_single_dimension_issues_one_reduction(self):
        self.assertEqual(self._calls(moe_ep_size=1, moe_tp_size=4), ["moe_tp"])
        self.assertEqual(self._calls(moe_ep_size=4, moe_tp_size=1), ["ep"])

    def test_skipped_when_deferred_to_fusion(self):
        self.assertEqual(self._calls(moe_ep_size=2, moe_tp_size=2, skip=True), [])


class TestFuseMlpAllReduceGate(unittest.TestCase):
    """Fusion is allowed only when one group covers the whole reduction.

    The fused residual+LN reduces over a single group. Hybrid EP+TP produces two
    reductions over disjoint groups; merging collapses them to one _TP reduction
    that the fused kernel can absorb. When merging does not apply
    (moe_dp_size > 1) there is no such group and fusion must stay off --
    otherwise the fused reduce covers half the peers and silently under-reduces.
    """

    def _should_fuse(self, *, moe_ep_size, moe_tp_size, moe_dp_size=1):
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )
        with (
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(
                moe_ep_size=moe_ep_size,
                moe_tp_size=moe_tp_size,
                moe_dp_size=moe_dp_size,
                tp_size=4,
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(), forward_batch
            )

    def test_hybrid_ep_tp_fuses_when_mergeable(self):
        self.assertTrue(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_hybrid_ep_tp_does_not_fuse_when_moe_dp_blocks_the_merge(self):
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2, moe_dp_size=2))

    def test_pure_tp_still_fuses(self):
        # moe_ep_size == 1: the whole post-experts reduction is the _MOE_TP one,
        # so a single fused all-reduce does cover every peer.
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        # moe_tp_size == 1: symmetric, the _MOE_EP reduce covers every peer.
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))


if __name__ == "__main__":
    unittest.main()
