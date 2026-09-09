"""Regression for DFLASH aux-hidden capture on Qwen4-Exp.

Qwen4-Exp keeps the residual stream in the hyper-connection layout
(hc_count * hidden_size) between layers, so captured aux hidden states
must be contracted back to hidden_size for the drafter; plain
hidden_size inputs (the first capture edge, before any HC mix) must
pass through untouched.
"""

import unittest

import torch
from torch import nn

from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4ExpDflashCapture(CustomTestCase):
    def _make_model(self, hc_count: int = 4, hidden_size: int = 3) -> Qwen4ExpModel:
        model = Qwen4ExpModel.__new__(Qwen4ExpModel)
        nn.Module.__init__(model)
        model.hc_count = hc_count
        model.hidden_size = hidden_size
        return model

    def test_dflash_contracts_hc_stream(self):
        model = self._make_model()

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        actual = model._prepare_aux_hidden_state(hidden_states, residual=None)
        expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3))

    def test_dflash_adds_residual_before_contracting(self):
        model = self._make_model()

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        residual = torch.full_like(hidden_states, 2)

        actual = model._prepare_aux_hidden_state(hidden_states, residual)
        expected = (hidden_states + residual).unflatten(-1, (4, -1)).mean(dim=-2)

        torch.testing.assert_close(actual, expected)

    def test_capture_passes_through_plain_hidden_size_stream(self):
        # First-layer input (before any hyper-connection mix) is hidden_size
        # wide and must be captured as-is.
        model = self._make_model()

        hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        actual = model._prepare_aux_hidden_state(hidden_states, residual=None)

        torch.testing.assert_close(actual, hidden_states)


if __name__ == "__main__":
    unittest.main()
