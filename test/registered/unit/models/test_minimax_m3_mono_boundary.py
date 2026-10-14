"""MiniMax-M3's mono launch publishes a complete contribution across the model's stage boundary."""

import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxM3MonoBoundary(CustomTestCase):
    def setUp(self):
        self.group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x * 4))
        self.stream = ResidualStream(torch.full((2, 4), 3.0))
        self.hidden = self.stream.record(
            UnreducedOutput(torch.ones(2, 4), group=self.group), PLAIN_ADD
        )
        self.batch = SimpleNamespace(residual_stream=self.stream)

    def layer(self, module_forward):
        layer = SimpleNamespace(
            layer_id=3,
            ffn_boundary=SimpleNamespace(
                declaration=SimpleNamespace(update=PLAIN_ADD),
                finish=Mock(side_effect=AssertionError("output already reduced")),
                complete_now=Mock(side_effect=AssertionError("output already reduced")),
            ),
            _forward_modules=module_forward,
        )
        layer._forward_mono = types.MethodType(
            MiniMaxM3DecoderLayer._forward_mono, layer
        )
        return layer

    def test_complete_output_and_eagle_capture_survive_kernel_buffer_reuse(self):
        scratch = torch.full((2, 4), 7.0)
        output = torch.full((2, 4), 11.0)
        residual = torch.full((2, 4), 7.0)

        def launch(step, layer_id, positions, hidden, previous_residual, batch, memo):
            # the pending contribution completed (all-reduced), its add left to the kernel
            torch.testing.assert_close(hidden, torch.full((2, 4), 4.0))
            torch.testing.assert_close(previous_residual, torch.full((2, 4), 3.0))
            self.assertEqual(layer_id, 3)
            return output, residual

        route = SimpleNamespace(
            kernel=SimpleNamespace(
                launch=Mock(side_effect=launch), capture_input=lambda: scratch
            ),
            step=SimpleNamespace(rows=2),
            cache={},
            checker=None,
        )
        layer = self.layer(Mock(side_effect=AssertionError("routed to the kernels")))
        captured = AuxHiddenStateList()
        result = MiniMaxM3DecoderLayer.forward(
            layer,
            None,
            self.hidden,
            self.batch,
            captured_last_layer_outputs=captured,
            mono_route=route,
        )
        self.group.all_reduce.assert_called_once()
        route.kernel.launch.assert_called_once()
        self.assertIs(result, output)
        self.assertIsNone(self.stream.pending.owed)

        # EAGLE keeps the pre-norm residual after the next launch reuses the buffer
        scratch.zero_()
        torch.testing.assert_close(captured[0], torch.full((2, 4), 7.0))

        def norm(hidden, previous_residual):
            combined = hidden + previous_residual
            return combined, combined

        result = residual_batch.final_norm(result, self.batch, norm)
        torch.testing.assert_close(result, torch.full((2, 4), 18.0))
        self.group.all_reduce.assert_called_once()

    def test_unrouted_layer_runs_the_modules(self):
        def modules(positions, hidden, batch, **kwargs):
            value, residual = self.stream.export(hidden)
            torch.testing.assert_close(value, torch.full((2, 4), 4.0))
            return self.stream.write(value + residual)

        module_forward = Mock(side_effect=modules)
        layer = self.layer(module_forward)
        result = MiniMaxM3DecoderLayer.forward(
            layer, None, self.hidden, self.batch, mono_route=None
        )
        torch.testing.assert_close(result, torch.full((2, 4), 7.0))
        module_forward.assert_called_once()
        self.group.all_reduce.assert_called_once()


if __name__ == "__main__":
    unittest.main()
