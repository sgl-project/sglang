"""Unit tests for the aiter all-reduce + RMSNorm fusion gate.

These cover ``LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer``,
specifically the AMD/aiter branch guards that disable the fused path under
DP attention or an expert-parallel A2A backend (e.g. mori). Without those
guards the fused custom all-reduce is invoked during CUDA graph capture in
those configs and crashes in ``custom_all_reduce.flush_graph_buffers``.

The gate is pure decision logic, so the test stubs out the module-level
dependencies and invokes the method on a minimal fake instance. No GPU or
distributed initialization is required.
"""

import types
import unittest
from contextlib import ExitStack
from unittest import mock

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _fake_self(*, mlp_mode=ScatterMode.TP_ATTN_FULL, is_last_layer=False, tp_size=8):
    """Minimal stand-in for a LayerCommunicator with the fields the gate reads."""
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=mlp_mode),
        is_last_layer=is_last_layer,
        _context=types.SimpleNamespace(tp_size=tp_size),
    )


def _fake_forward_batch(batch_size=8):
    return types.SimpleNamespace(
        input_ids=types.SimpleNamespace(shape=(batch_size,))
    )


class TestAiterAllreduceFusionGate(CustomTestCase):
    def _evaluate_gate(
        self,
        *,
        dp_attention,
        a2a_is_none,
        aiter_enabled=True,
        use_aiter=True,
        tp_world_size=8,
        mlp_mode=ScatterMode.TP_ATTN_FULL,
        is_last_layer=False,
        tp_size=8,
    ):
        """Run the gate with the aiter branch isolated (flashinfer forced off)."""
        server_args = types.SimpleNamespace(
            enable_aiter_allreduce_fusion=aiter_enabled
        )
        a2a_backend = types.SimpleNamespace(is_none=lambda: a2a_is_none)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(comm, "is_enable_moe_cp_allgather", lambda: False)
            )
            stack.enter_context(
                mock.patch.object(
                    comm,
                    "get_attn_tp_context",
                    lambda: types.SimpleNamespace(input_scattered=False),
                )
            )
            # Force the NVIDIA/flashinfer term off so the aiter branch decides.
            stack.enter_context(
                mock.patch.object(
                    comm, "apply_flashinfer_allreduce_fusion", lambda batch_size: False
                )
            )
            stack.enter_context(mock.patch.object(comm, "_use_aiter", use_aiter))
            stack.enter_context(
                mock.patch.object(
                    comm, "get_tensor_model_parallel_world_size", lambda: tp_world_size
                )
            )
            stack.enter_context(
                mock.patch.object(comm, "get_global_server_args", lambda: server_args)
            )
            stack.enter_context(
                mock.patch.object(
                    comm, "is_dp_attention_enabled", lambda: dp_attention
                )
            )
            stack.enter_context(
                mock.patch.object(comm, "get_moe_a2a_backend", lambda: a2a_backend)
            )

            fake_self = _fake_self(
                mlp_mode=mlp_mode, is_last_layer=is_last_layer, tp_size=tp_size
            )
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                fake_self, _fake_forward_batch()
            )

    def test_dense_tp_fuses(self):
        # Baseline supported path: dense TP, no DP attention, no EP backend.
        self.assertTrue(
            self._evaluate_gate(dp_attention=False, a2a_is_none=True)
        )

    def test_dp_attention_disables_fusion(self):
        # The fix: DP attention has no dense TP all-reduce to fuse.
        self.assertFalse(
            self._evaluate_gate(dp_attention=True, a2a_is_none=True)
        )

    def test_ep_backend_disables_fusion(self):
        # The fix: with an EP A2A backend (e.g. mori) the reduction lives in
        # combine(), not a TP all-reduce.
        self.assertFalse(
            self._evaluate_gate(dp_attention=False, a2a_is_none=False)
        )

    def test_dp_attention_and_ep_disables_fusion(self):
        # The crashing config from the TP8+EP8+mori repro.
        self.assertFalse(
            self._evaluate_gate(dp_attention=False, a2a_is_none=False)
        )
        self.assertFalse(
            self._evaluate_gate(dp_attention=True, a2a_is_none=False)
        )

    def test_flag_off_disables_fusion(self):
        # Sanity: the gate still respects the opt-in flag on the dense path.
        self.assertFalse(
            self._evaluate_gate(
                dp_attention=False, a2a_is_none=True, aiter_enabled=False
            )
        )

    def test_last_layer_disables_fusion(self):
        self.assertFalse(
            self._evaluate_gate(
                dp_attention=False, a2a_is_none=True, is_last_layer=True
            )
        )

    def test_tp1_disables_fusion(self):
        self.assertFalse(
            self._evaluate_gate(dp_attention=False, a2a_is_none=True, tp_size=1)
        )


if __name__ == "__main__":
    unittest.main()
