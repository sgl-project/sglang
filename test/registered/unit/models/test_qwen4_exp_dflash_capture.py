"""Regression for DFLASH aux-hidden capture on Qwen4-Exp.

Qwen4-Exp keeps the residual stream in the hyper-connection layout
(hc_count * hidden_size) between layers, so captured aux hidden states
must be contracted back to hidden_size for the drafter. The contraction
must use the captured layer's own learned gate
(attn_hyper_connection.mix), not a uniform hc_contract average: hardware
checks on 6413 positions showed hc_contract matches the reference hidden
states at cos 0.000, while the learned mix reaches 0.23-0.89.
"""

import unittest

import torch
from torch import nn

from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LearnedMixStub(nn.Module):
    """CPU stand-in for GatedResidual.mix with non-uniform branch weights.

    Shape-faithful to the real mix contract: hc-wide input in, mixed
    hidden_size output first, residual tuple second.
    """

    def __init__(self, hc_count: int, hidden_size: int):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        # Deliberately non-uniform so a uniform-average contraction (the old
        # hc_contract behavior) cannot pass these tests.
        self.branch_weights = nn.Parameter(
            torch.arange(1, hc_count + 1, dtype=torch.float32)
            .reshape(1, hc_count, 1)
            .repeat(1, 1, hidden_size)
        )

    def mix(self, x: torch.Tensor):
        branches = x.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (branches * self.branch_weights).mean(dim=-2)
        return mixed, (x, x)


class _FakeLayer(nn.Module):
    def __init__(self, attn_hyper_connection: nn.Module):
        super().__init__()
        self.attn_hyper_connection = attn_hyper_connection


class TestQwen4ExpDflashCapture(CustomTestCase):
    def _make_model(self, hc_count: int = 4, hidden_size: int = 3) -> Qwen4ExpModel:
        model = Qwen4ExpModel.__new__(Qwen4ExpModel)
        nn.Module.__init__(model)
        model.hc_count = hc_count
        model.hidden_size = hidden_size
        return model

    def _make_layer(self, hc_count: int = 4, hidden_size: int = 3) -> nn.Module:
        return _FakeLayer(_LearnedMixStub(hc_count, hidden_size))

    def test_dflash_uses_layer_learned_mix(self):
        model = self._make_model()
        layer = self._make_layer()

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        actual = model._prepare_aux_hidden_state(layer, hidden_states, None)
        expected = layer.attn_hyper_connection.mix(hidden_states)[0]

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3))
        # The learned gate is not a uniform average over the hc branches.
        uniform = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)
        self.assertGreater((actual - uniform).abs().max().item(), 1e-3)

    def test_dflash_adds_residual_before_mix(self):
        model = self._make_model()
        layer = self._make_layer()

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        residual = torch.full_like(hidden_states, 2)

        actual = model._prepare_aux_hidden_state(layer, hidden_states, residual)
        expected = layer.attn_hyper_connection.mix(hidden_states + residual)[0]

        torch.testing.assert_close(actual, expected)

    def test_capture_tiles_plain_hidden_size_stream_before_mix(self):
        # First-layer input (before any hyper-connection mix) is hidden_size
        # wide; mirror _prepare_qwen4_exp_attn by tiling into the
        # hyper-connection layout before the learned mix instead of passing
        # the live tensor through.
        model = self._make_model()
        layer = self._make_layer()

        hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        actual = model._prepare_aux_hidden_state(layer, hidden_states, None)
        tiled = torch.cat([hidden_states for _ in range(4)], dim=-1)
        expected = layer.attn_hyper_connection.mix(tiled)[0]

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
