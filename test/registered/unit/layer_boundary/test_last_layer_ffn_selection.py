"""Shared terminal FFN selection is architecture-independent and preserves residuals."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.layer_boundary.contracts import StageKind
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTerminalFfnSelection(CustomTestCase):
    def test_only_enabled_terminal_ffn_selects_rows(self):
        for kind, terminal, enabled in (
            (StageKind.FFN, True, True),
            (StageKind.FFN, False, True),
            (StageKind.ATTENTION, True, True),
            (StageKind.FFN, True, False),
        ):
            with self.subTest(kind=kind, terminal=terminal, enabled=enabled):
                x = torch.arange(32.0).reshape(8, 4)
                residual = x + 100
                hidden, stream = ResidualStream.from_handoff(x, residual, PLAIN_ADD)
                fb = SimpleNamespace(
                    residual_stream=stream,
                    last_layer_ffn_indices=torch.tensor([2, 7]) if enabled else None,
                    last_layer_ffn_pruned=False,
                )
                boundary = StageBoundary.__new__(StageBoundary)
                boundary.declaration = SimpleNamespace(kind=kind, terminal=terminal)
                prepare = Mock(
                    side_effect=lambda value, state, batch, **kw: (value, state)
                )
                boundary._prepare_input = prepare
                boundary._prepare_attention = prepare
                out = boundary.prepare(hidden, fb)
                actual, prior = fb.residual_stream.export(out)
                compact = kind is StageKind.FFN and terminal and enabled
                idx = [2, 7] if compact else list(range(8))
                torch.testing.assert_close(actual, x[idx], rtol=0, atol=0)
                torch.testing.assert_close(prior, residual[idx], rtol=0, atol=0)
                self.assertEqual(fb.last_layer_ffn_pruned, compact)
                prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
