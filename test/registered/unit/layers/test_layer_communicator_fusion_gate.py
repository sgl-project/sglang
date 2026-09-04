import types
import unittest
from unittest.mock import patch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import (
    CommunicateContext,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    enable_moe_fully_dp,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator():
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=ScatterMode.TP_ATTN_FULL),
        is_last_layer=False,
        _context=types.SimpleNamespace(tp_size=4),
    )


class TestFuseMlpAllReduceGate(CustomTestCase):
    """Hybrid EP+TP must not fuse the post-experts all-reduce away.

    The fused residual+LN reduces over a single group, but with moe_ep_size > 1
    and moe_tp_size > 1 the post-experts reduction spans two disjoint groups
    (_MOE_EP then _MOE_TP) and should_skip_post_experts_all_reduce() drops both
    once fusion is published. The result is activations reduced over only half
    the peers -- wrong output, no crash. Observed as garbage completions on
    Qwen3-30B-A3B with --tp-size 4 --ep-size 2.
    """

    def _should_fuse(self, *, moe_ep_size, moe_tp_size):
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
                moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size, tp_size=4
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(), forward_batch
            )

    def test_hybrid_ep_tp_does_not_fuse(self):
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))


class TestFullReplicaMoE(CustomTestCase):
    def _topology(self, **overrides):
        topology = dict(
            tp_rank=0,
            tp_size=2,
            pp_size=1,
            dcp_enabled=False,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_dp_size=2,
            attn_cp_rank=0,
            attn_cp_size=1,
            moe_ep_size=1,
            moe_tp_size=1,
            moe_dp_size=2,
            moe_dense_tp_size=1,
            dwdp_size=1,
        )
        topology.update(overrides)
        return topology

    def test_full_replica_sparse_mlp_stays_local(self):
        backend = types.SimpleNamespace(is_none=lambda: True)
        mode_args = dict(
            num_layers=2,
            layer_id=0,
            is_layer_sparse=True,
            is_previous_layer_sparse=None,
            is_next_layer_sparse=True,
        )
        with (
            patch.object(comm, "is_dp_attention_enabled", return_value=True),
            patch.object(comm, "get_moe_a2a_backend", return_value=backend),
            patch.object(
                comm,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(comm, "is_mla_prefill_cp_enabled", return_value=False),
            patch.object(
                comm,
                "get_parallel",
                return_value=types.SimpleNamespace(**self._topology()),
            ),
        ):
            self.assertTrue(enable_moe_fully_dp())
            self.assertIs(
                LayerScatterModes.init_new(**mode_args).mlp_mode,
                ScatterMode.SCATTERED,
            )

        with (
            patch.object(comm, "is_dp_attention_enabled", return_value=True),
            patch.object(comm, "get_moe_a2a_backend", return_value=backend),
            patch.object(
                comm,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(comm, "is_mla_prefill_cp_enabled", return_value=False),
            patch.object(
                comm,
                "get_parallel",
                return_value=types.SimpleNamespace(**self._topology(moe_tp_size=2)),
            ),
        ):
            self.assertFalse(enable_moe_fully_dp())
            self.assertIs(
                LayerScatterModes.init_new(**mode_args).mlp_mode,
                ScatterMode.FULL,
            )

    def test_full_replica_moe_full_width_is_defined(self):
        backend = types.SimpleNamespace(is_none=lambda: True)
        with (
            patch.object(comm, "is_dp_attention_enabled", return_value=True),
            patch.object(comm, "get_moe_a2a_backend", return_value=backend),
            patch.object(comm, "get_moe_cp_size", return_value=2),
            patch.object(
                comm,
                "get_parallel",
                return_value=types.SimpleNamespace(**self._topology()),
            ),
        ):
            context = CommunicateContext.init_new()

        self.assertEqual(context.process_group_sizes[ScatterMode.FULL], 2)
        self.assertEqual(context.process_group_sizes[ScatterMode.MOE_FULL], 2)

    def test_invalid_non_replica_moe_width_fails_early(self):
        backend = types.SimpleNamespace(is_none=lambda: True)
        with (
            patch.object(comm, "is_dp_attention_enabled", return_value=True),
            patch.object(comm, "get_moe_a2a_backend", return_value=backend),
            patch.object(comm, "get_moe_cp_size", return_value=2),
            patch.object(
                comm,
                "get_parallel",
                return_value=types.SimpleNamespace(**self._topology(moe_tp_size=2)),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "must be divisible"):
                CommunicateContext.init_new()


if __name__ == "__main__":
    unittest.main()
