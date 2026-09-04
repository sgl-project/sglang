"""Regression for DFLASH aux-hidden capture on mHC models.

GLM-5.3-Flash runs with mhc=True. MHCLayerCommunicator folds the residual
into the widened hidden state and returns residual=None, so CUDA-graph
capture used to crash on `hidden_states + residual`. DFLASH also has to
contract that widened state back to the draft hidden size; skipping the
contract is a silent shape/quality bug the crash-guard alone would miss.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextDflashCapture(CustomTestCase):
    def test_dflash_contracts_mhc_hidden_state_without_residual(self):
        model = Glm5NextModel.__new__(Glm5NextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(mhc=True, hc_mult=4)
        model.dflash_capture = True

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        actual = model._prepare_aux_hidden_state(hidden_states, None)
        expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
