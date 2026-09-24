import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.communicator import LayerCommunicator
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_communicator(*, fuse, reduce_scatter, cls=LayerCommunicator):
    """A communicator whose decisions and postprocess are stubbed, built
    without the process-wide parallel state."""
    communicator = cls.__new__(cls)
    if cls is LayerCommunicator:
        communicator.should_fuse_mlp_allreduce_with_next_layer = MagicMock(
            return_value=fuse
        )
    communicator.should_use_reduce_scatter = MagicMock(return_value=reduce_scatter)
    communicator.postprocess_layer = MagicMock(
        side_effect=lambda hidden_states, residual, forward_batch: (
            hidden_states + 1,
            residual,
        )
    )
    return communicator


def published_flags():
    forward = get_forward()
    return forward.fuse_mlp_allreduce, forward.mlp_reduce_scatter


class TestFfnExit(CustomTestCase):
    def setUp(self):
        self.hidden_states = torch.ones(3, 4)
        self.residual = torch.zeros(3, 4)
        self.forward_batch = object()

    def run_exit(self, communicator):
        with communicator.ffn_exit(self.forward_batch) as ffn_exit:
            seen = published_flags()
            hidden_states = self.hidden_states * 2
        return seen, ffn_exit.finish(hidden_states, self.residual)

    def test_reduction_left_to_next_layer_is_marked(self):
        communicator = make_communicator(fuse=True, reduce_scatter=False)
        seen, (hidden_states, residual) = self.run_exit(communicator)
        self.assertEqual(seen, (True, False))
        self.assertTrue(hidden_states._sglang_needs_allreduce_fusion)
        self.assertIs(residual, self.residual)
        communicator.postprocess_layer.assert_not_called()

    def test_postprocess_completes_other_exits(self):
        for reduce_scatter in (False, True):
            with self.subTest(reduce_scatter=reduce_scatter):
                communicator = make_communicator(
                    fuse=False, reduce_scatter=reduce_scatter
                )
                seen, (hidden_states, residual) = self.run_exit(communicator)
                self.assertEqual(seen, (False, reduce_scatter))
                self.assertFalse(
                    getattr(hidden_states, "_sglang_needs_allreduce_fusion", False)
                )
                communicator.postprocess_layer.assert_called_once()
                torch.testing.assert_close(hidden_states, self.hidden_states * 2 + 1)
                self.assertIs(residual, self.residual)

    def test_finish_applies_the_decision_the_ffn_saw(self):
        """finish() applies the decision published to the FFN, even when the
        decision methods would answer differently by then."""
        for fuse in (True, False):
            with self.subTest(fuse=fuse):
                communicator = make_communicator(fuse=fuse, reduce_scatter=False)
                with communicator.ffn_exit(self.forward_batch) as ffn_exit:
                    seen = published_flags()
                    decide = communicator.should_fuse_mlp_allreduce_with_next_layer
                    decide.return_value = not fuse
                    hidden_states = self.hidden_states * 2
                hidden_states, _ = ffn_exit.finish(hidden_states, self.residual)
                self.assertEqual(seen, (fuse, False))
                self.assertEqual(
                    getattr(hidden_states, "_sglang_needs_allreduce_fusion", False),
                    fuse,
                )
                self.assertEqual(communicator.postprocess_layer.called, not fuse)

    def test_flags_are_restored_after_the_ffn(self):
        before = published_flags()
        self.run_exit(make_communicator(fuse=True, reduce_scatter=True))
        self.assertEqual(published_flags(), before)

    def test_subclass_decisions_are_used(self):
        class NeverFuses(LayerCommunicator):
            def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
                return False

        communicator = make_communicator(
            fuse=True, reduce_scatter=False, cls=NeverFuses
        )
        seen, _ = self.run_exit(communicator)
        self.assertEqual(seen, (False, False))
        communicator.postprocess_layer.assert_called_once()

    def test_deferral_implies_fusion_and_passes_the_handoff_through(self):
        """A deferring communicator publishes both flags, and a non-tensor
        handoff leaves finish() untouched for the next layer's input norm."""

        class Defers(LayerCommunicator):
            def should_defer_moe_finalize(self, forward_batch, m=None):
                return True

            def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
                return False

        communicator = make_communicator(fuse=False, reduce_scatter=False, cls=Defers)
        handoff = object()
        with communicator.ffn_exit(self.forward_batch) as ffn_exit:
            seen = published_flags() + (get_forward().defer_moe_finalize,)
        self.assertEqual(seen, (True, False, True))
        self.assertEqual(
            ffn_exit.finish(handoff, self.residual), (handoff, self.residual)
        )
        communicator.postprocess_layer.assert_not_called()
        self.assertFalse(get_forward().defer_moe_finalize)

        # The MoE may decline per forward; a tensor then takes the fused exit.
        hidden_states, _ = ffn_exit.finish(self.hidden_states * 2, self.residual)
        self.assertTrue(hidden_states._sglang_needs_allreduce_fusion)

    def test_compiles_without_graph_breaks(self):
        communicator = LayerCommunicator.__new__(LayerCommunicator)
        communicator.should_fuse_mlp_allreduce_with_next_layer = lambda forward_batch: (
            True
        )
        communicator.should_use_reduce_scatter = lambda forward_batch: False
        forward_batch = object()

        def layer(hidden_states, residual):
            with communicator.ffn_exit(forward_batch) as ffn_exit:
                if get_forward().fuse_mlp_allreduce:
                    hidden_states = hidden_states * 2
            return ffn_exit.finish(hidden_states, residual)

        torch._dynamo.reset()
        compiled = torch.compile(layer, backend="eager", fullgraph=True)
        hidden_states, residual = compiled(self.hidden_states, self.residual)
        torch.testing.assert_close(hidden_states, self.hidden_states * 2)
        self.assertIs(residual, self.residual)


if __name__ == "__main__":
    unittest.main()
