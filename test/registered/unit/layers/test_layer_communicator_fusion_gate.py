import contextlib
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.layers.moe import (
    can_merge_post_experts_all_reduce,
    deferred_post_experts_all_reduce,
    post_experts_all_reduce,
)
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _fake_communicator(mlp_mode=ScatterMode.TP_ATTN_FULL):
    communicator = LayerCommunicator.__new__(LayerCommunicator)
    communicator._speculative_algo = None
    communicator.layer_scatter_modes = types.SimpleNamespace(mlp_mode=mlp_mode)
    communicator.is_last_layer = False
    communicator._context = types.SimpleNamespace(tp_size=4)
    return communicator


@contextlib.contextmanager
def _recorded_all_reduces(called, *, moe_ep_size, moe_tp_size, moe_dp_size):
    """Log which group each all-reduce helper reduces over, under a fixed topo."""

    def record(name):
        return lambda x: called.append(name) or x

    with (
        patch(
            "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
            side_effect=record("tp"),
        ),
        patch(
            "sglang.srt.distributed.communication_op.moe_expert_parallel_all_reduce",
            side_effect=record("ep"),
        ),
        patch(
            "sglang.srt.distributed.communication_op.moe_tensor_model_parallel_all_reduce",
            side_effect=record("moe_tp"),
        ),
        get_parallel().override(
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=moe_dp_size,
            tp_size=moe_ep_size * moe_tp_size * moe_dp_size,
        ),
    ):
        yield


class TestPostExpertsAllReduceMerge(CustomTestCase):
    """The two post-experts reductions collapse into one _TP reduction.

    _MOE_EP and _MOE_TP are orthogonal subgroups of _TP, so with
    moe_dp_size == 1 reducing over each in turn equals one _TP reduction --
    one collective instead of two. With moe_dp_size > 1 they cover only part of
    _TP and merging would sum across DP replicas, which hold different tokens.
    """

    def _calls(self, *, moe_ep_size, moe_tp_size, moe_dp_size=1, skip=False):
        called = []
        with (
            patch.object(
                moe_utils, "should_skip_post_experts_all_reduce", return_value=skip
            ),
            _recorded_all_reduces(
                called,
                moe_ep_size=moe_ep_size,
                moe_tp_size=moe_tp_size,
                moe_dp_size=moe_dp_size,
            ),
        ):
            post_experts_all_reduce(torch.zeros(2, 2))
        return called

    def test_hybrid_issues_one_tp_reduction(self):
        self.assertEqual(self._calls(moe_ep_size=2, moe_tp_size=2), ["tp"])

    def test_moe_dp_keeps_the_two_step_form(self):
        # Server args reject moe_ep_size > 1 together with moe_tp_size > 1 and
        # moe_dp_size > 1 (they force ep_size * moe_dp_size == tp_size), so this
        # pins the guard rather than a topology that can be launched today.
        self.assertEqual(
            self._calls(moe_ep_size=2, moe_tp_size=2, moe_dp_size=2), ["ep", "moe_tp"]
        )

    def test_single_dimension_issues_one_reduction(self):
        self.assertEqual(self._calls(moe_ep_size=1, moe_tp_size=4), ["moe_tp"])
        self.assertEqual(self._calls(moe_ep_size=4, moe_tp_size=1), ["ep"])

    def test_skipped_when_deferred_to_fusion(self):
        self.assertEqual(self._calls(moe_ep_size=2, moe_tp_size=2, skip=True), [])


class TestDeferredPostExpertsAllReduce(CustomTestCase):
    """The inline fallback must reduce over the same peers the fused kernel would.

    _MOE_TP holds a single rank under pure EP, so reducing over it there is a
    no-op that drops the deferred reduction instead of performing it.
    """

    def _calls(self, *, moe_ep_size, moe_tp_size, moe_dp_size=1):
        called = []

        def group(name):
            return types.SimpleNamespace(all_reduce=lambda x: called.append(name) or x)

        parallel = types.SimpleNamespace(
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=moe_dp_size,
            tp_group=group("tp"),
            moe_ep_group=group("ep"),
            moe_tp_group=group("moe_tp"),
        )
        with patch.object(moe_utils, "get_parallel", return_value=parallel):
            deferred_post_experts_all_reduce(torch.zeros(2, 2))
        return called

    def test_hybrid_reduces_over_tp(self):
        self.assertEqual(self._calls(moe_ep_size=2, moe_tp_size=2), ["tp"])

    def test_pure_ep_reduces_over_ep(self):
        self.assertEqual(self._calls(moe_ep_size=4, moe_tp_size=1), ["ep"])

    def test_pure_tp_reduces_over_moe_tp(self):
        self.assertEqual(self._calls(moe_ep_size=1, moe_tp_size=4), ["moe_tp"])

    def test_moe_dp_reduces_over_moe_tp(self):
        self.assertEqual(
            self._calls(moe_ep_size=1, moe_tp_size=2, moe_dp_size=2), ["moe_tp"]
        )


class TestCanMergePostExpertsAllReduce(CustomTestCase):
    def _can_merge(self, *, moe_ep_size, moe_tp_size, moe_dp_size=1):
        with get_parallel().override(
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=moe_dp_size,
            tp_size=moe_ep_size * moe_tp_size * moe_dp_size,
        ):
            return can_merge_post_experts_all_reduce()

    def test_hybrid_ep_tp_merges(self):
        self.assertTrue(self._can_merge(moe_ep_size=2, moe_tp_size=2))

    def test_single_dimension_does_not_merge(self):
        self.assertFalse(self._can_merge(moe_ep_size=1, moe_tp_size=4))
        self.assertFalse(self._can_merge(moe_ep_size=4, moe_tp_size=1))


class TestResolveFusionGroup(CustomTestCase):
    """EP2/MoE-TP2/DP1 (e.g. DeepSeek-V4-Flash with --tp-size 4 --ep-size 2) must
    resolve to the _TP group with world_size=4 and the TP rank."""

    def _resolve(self, *, moe_ep_size, moe_tp_size, moe_dp_size=1, tp_rank=0):
        from sglang.srt.layers.flashinfer_comm_fusion import (
            resolve_fusion_group,
            resolve_fusion_world_size,
        )

        tp_size = moe_ep_size * moe_tp_size * moe_dp_size
        fake_tp_group = MagicMock(
            name="tp_group", world_size=tp_size, rank_in_group=tp_rank
        )
        fake_ep_group = MagicMock(
            name="ep_group",
            world_size=moe_ep_size,
            rank_in_group=tp_rank % moe_ep_size,
        )
        fake_moe_tp_group = MagicMock(
            name="moe_tp_group",
            world_size=moe_tp_size,
            rank_in_group=tp_rank % moe_tp_size,
        )

        with get_parallel().override(
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=moe_dp_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            moe_ep_rank=tp_rank % moe_ep_size,
            moe_tp_rank=tp_rank % moe_tp_size,
            tp_group=fake_tp_group,
            moe_ep_group=fake_ep_group,
            moe_tp_group=fake_moe_tp_group,
        ):
            ws = resolve_fusion_world_size(use_attn_tp_group=False)
            group_tuple = resolve_fusion_group(use_attn_tp_group=False)
        return ws, group_tuple, (fake_tp_group, fake_ep_group, fake_moe_tp_group)

    def test_hybrid_ep2_tp2_dp1_resolves_to_tp_ws4(self):
        # EP2/MoE-TP2/DP1 (DeepSeek-V4-Flash on 4 GPUs): workspace must sit on
        # _TP (ws=4) so the fused kernel reduces over all 4 peers.
        ws, (size, rank, group), (tp_grp, ep_grp, moe_tp_grp) = self._resolve(
            moe_ep_size=2, moe_tp_size=2, moe_dp_size=1, tp_rank=3
        )
        self.assertEqual(ws, 4)
        self.assertEqual(size, 4)
        self.assertEqual(rank, 3)
        self.assertIs(group, tp_grp)

    def test_pure_ep_resolves_to_ep_group(self):
        ws, (size, rank, group), (tp_grp, ep_grp, moe_tp_grp) = self._resolve(
            moe_ep_size=4, moe_tp_size=1, moe_dp_size=1, tp_rank=2
        )
        self.assertEqual(ws, 4)
        self.assertEqual(size, 4)
        self.assertIs(group, ep_grp)

    def test_pure_tp_resolves_to_moe_tp_group(self):
        ws, (size, rank, group), (tp_grp, ep_grp, moe_tp_grp) = self._resolve(
            moe_ep_size=1, moe_tp_size=4, moe_dp_size=1, tp_rank=1
        )
        self.assertEqual(ws, 4)
        self.assertEqual(size, 4)
        self.assertIs(group, moe_tp_grp)


class TestFuseMlpAllReduceGate(CustomTestCase):
    """Fusion is allowed only when one group covers the whole reduction.

    The fused residual+LN reduces over a single group. Hybrid EP+TP produces two
    reductions over disjoint groups; merging collapses them to one _TP reduction
    that the fused kernel can absorb. When merging does not apply
    (moe_dp_size > 1) there is no such group and fusion must stay off --
    otherwise the fused reduce covers half the peers and silently under-reduces.
    """

    def _should_fuse(
        self,
        *,
        moe_ep_size,
        moe_tp_size,
        moe_dp_size=1,
        mlp_mode=ScatterMode.TP_ATTN_FULL,
    ):
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
                tp_size=moe_ep_size * moe_tp_size * moe_dp_size,
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(mlp_mode), forward_batch
            )

    def test_hybrid_ep_tp_fuses_when_mergeable(self):
        self.assertTrue(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_hybrid_ep_tp_does_not_fuse_when_moe_dp_blocks_the_merge(self):
        # Same caveat as test_moe_dp_keeps_the_two_step_form: unreachable today,
        # kept so a future relaxation cannot silently re-enable fusion over a
        # reduction that no single group covers.
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2, moe_dp_size=2))

    def test_pure_tp_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))

    def test_moe_full_layer_does_not_fuse(self):
        # Fusion skips postprocess_layer, which holds the CP scatter; a dense
        # MOE_FULL layer (moe_dp_size == attn_cp_size) is not caught by the
        # is_enable_moe_cp_allgather gate.
        self.assertFalse(
            self._should_fuse(
                moe_ep_size=1, moe_tp_size=4, mlp_mode=ScatterMode.MOE_FULL
            )
        )


class TestDeferFfnReduction(CustomTestCase):
    """The FFN leaves its sum to the next layer whenever the next layer runs the
    all-reduce the FFN would have: fused when the kernel takes the batch, and
    otherwise the same full-precision TP all-reduce."""

    def _should_defer(
        self,
        *,
        fused=False,
        dp_attention=False,
        a2a_none=True,
        output_complete=False,
        quant_communications=False,
        reduce_scatter=False,
        is_last_layer=False,
        batch_size=8,
        shared_expert_tp1=False,
        lora=False,
        tp_group=True,
        global_tokens=8,
        scatters_to_local_tokens=True,
        step=None,
        sp_active=False,
    ):
        communicator = _fake_communicator()
        communicator.is_last_layer = is_last_layer
        communicator.should_use_reduce_scatter = lambda forward_batch: reduce_scatter
        communicator._postprocess_scatters_to_local_tokens = scatters_to_local_tokens
        communicator._sp_variant = object() if sp_active else None
        communicator.allow_reduce_scatter = True
        communicator.layer_scatter_modes.is_layer_sparse = True
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(batch_size,)),
            global_dp_buffer_len=global_tokens,
        )
        with (
            patch.object(
                comm, "_reduce_and_redistribute_output_step", return_value=step
            ),
            get_forward().scoped(sp_active=sp_active),
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=fused),
            patch.object(comm, "_use_aiter", False),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            patch.object(comm, "is_dp_attention_enabled", return_value=dp_attention),
            patch.object(
                comm,
                "get_moe_a2a_backend",
                return_value=types.SimpleNamespace(is_none=lambda: a2a_none),
            ),
            patch.object(
                comm, "post_experts_output_is_complete", return_value=output_complete
            ),
            patch.object(
                comm,
                "get_exec",
                return_value=types.SimpleNamespace(
                    comm=types.SimpleNamespace(
                        enable_quant_communications=quant_communications
                    )
                ),
            ),
            patch.object(
                comm.envs.SGLANG_SHARED_EXPERT_TP1,
                "get",
                return_value=shared_expert_tp1,
            ),
            patch.object(
                comm,
                "get_lora",
                return_value=types.SimpleNamespace(enable_lora=lora),
            ),
            patch.object(
                comm, "_deferred_reduction_runs_on_the_tp_group", return_value=tp_group
            ),
            get_parallel().override(
                moe_ep_size=1, moe_tp_size=4, moe_dp_size=1, tp_size=4
            ),
        ):
            return communicator.should_defer_ffn_reduction(forward_batch)

    def test_defers_whether_or_not_the_fused_kernel_takes_the_batch(self):
        self.assertTrue(self._should_defer(fused=True))
        self.assertTrue(self._should_defer(fused=False))

    def test_keeps_the_reduction_when_the_next_layer_would_run_a_different_one(
        self,
    ):
        for condition in (
            dict(a2a_none=False),
            dict(output_complete=True),
            dict(quant_communications=True),
            dict(reduce_scatter=True),
            dict(is_last_layer=True),
            dict(batch_size=0),
            dict(shared_expert_tp1=True),
            dict(lora=True),
            dict(tp_group=False),
        ):
            with self.subTest(**condition):
                self.assertFalse(self._should_defer(**condition))


class TestDeferFfnReductionUnderAttentionDp(CustomTestCase):
    """Under attention DP the next layer's input runs the all-reduce and then the
    scatter back to this rank's tokens that postprocess would have run."""

    _should_defer = TestDeferFfnReduction._should_defer

    def test_defers_when_postprocess_would_only_scatter(self):
        self.assertTrue(self._should_defer(dp_attention=True))

    def test_every_rank_decides_from_all_ranks_tokens(self):
        self.assertTrue(self._should_defer(dp_attention=True, batch_size=0))
        self.assertFalse(self._should_defer(dp_attention=True, global_tokens=0))

    def test_keeps_what_the_next_layer_input_cannot_run(self):
        for name, condition in (
            ("reduce-scatter", dict(step=comm._reduce_and_redistribute_output_varlen)),
            (
                "MAX_LEN reduce-scatter",
                dict(step=comm._reduce_and_redistribute_output_max_len),
            ),
            ("other postprocess", dict(scatters_to_local_tokens=False)),
            ("LayerNorm SP", dict(sp_active=True)),
        ):
            with self.subTest(name):
                self.assertFalse(self._should_defer(dp_attention=True, **condition))


class TestDeferredReductionGroup(CustomTestCase):
    """The unfused completion runs on the same group object the FFN would have
    reduced over, or the FFN keeps its reduction."""

    def _runs_on_tp(
        self, *, moe_ep_size, moe_tp_size, ep_is_tp=True, moe_tp_is_tp=True
    ):
        tp_group = object()
        parallel = types.SimpleNamespace(
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=1,
            tp_group=tp_group,
            moe_ep_group=tp_group if ep_is_tp else object(),
            moe_tp_group=tp_group if moe_tp_is_tp else object(),
        )
        with (
            patch.object(comm, "get_parallel", return_value=parallel),
            patch.object(moe_utils, "get_parallel", return_value=parallel),
        ):
            return comm._deferred_reduction_runs_on_the_tp_group()

    def test_pure_tp_and_pure_ep_reduce_over_the_tp_group(self):
        self.assertTrue(self._runs_on_tp(moe_ep_size=1, moe_tp_size=4))
        self.assertTrue(self._runs_on_tp(moe_ep_size=4, moe_tp_size=1))

    def test_a_separate_group_keeps_the_reduction_in_the_ffn(self):
        self.assertFalse(
            self._runs_on_tp(moe_ep_size=1, moe_tp_size=2, moe_tp_is_tp=False)
        )
        self.assertFalse(self._runs_on_tp(moe_ep_size=4, moe_tp_size=1, ep_is_tp=False))

    def test_hybrid_ep_tp_keeps_the_reduction_in_the_ffn(self):
        self.assertFalse(self._runs_on_tp(moe_ep_size=2, moe_tp_size=2))


if __name__ == "__main__":
    unittest.main()
