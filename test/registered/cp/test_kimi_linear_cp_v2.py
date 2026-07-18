import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.kimi_linear import KimiLinearCPV2LayerCommunicator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingStrategy:
    def __init__(self):
        self.gather_calls = 0

    def gather_hidden_states(self, hidden_states, forward_batch, stream=None):
        del forward_batch, stream
        self.gather_calls += 1
        return hidden_states + 10


class TestKimiLinearCPV2LayerCommunicator(CustomTestCase):
    def test_first_kda_layer_gathers_model_entry_shard(self):
        strategy = _RecordingStrategy()
        communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=True,
            previous_is_kda_layer=None,
        )
        hidden_states = torch.arange(4).view(2, 2)

        with (
            patch(
                "sglang.srt.layers.cp.kimi_linear.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.cp.kimi_linear.get_cp_strategy",
                return_value=strategy,
            ),
        ):
            output, residual = communicator.prepare_attn(
                hidden_states,
                None,
                SimpleNamespace(),
            )

        self.assertEqual(strategy.gather_calls, 1)
        torch.testing.assert_close(output, hidden_states + 10)
        self.assertIsNone(residual)


if __name__ == "__main__":
    unittest.main()
