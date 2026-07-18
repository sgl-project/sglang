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
        self.shard_calls = 0

    def gather_hidden_states(self, hidden_states, forward_batch, stream=None):
        del forward_batch, stream
        self.gather_calls += 1
        return hidden_states + 10

    def shard_hidden_states(self, hidden_states, forward_batch):
        del forward_batch
        self.shard_calls += 1
        return hidden_states[::2]


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

    def test_kda_to_mla_shards_hidden_states_and_residual(self):
        strategy = _RecordingStrategy()
        communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=False,
            previous_is_kda_layer=True,
        )
        hidden_states = torch.arange(8).view(4, 2)
        residual = hidden_states + 100

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
            output, output_residual = communicator.prepare_attn(
                hidden_states,
                residual,
                SimpleNamespace(),
            )

        self.assertEqual(strategy.shard_calls, 2)
        torch.testing.assert_close(output, hidden_states[::2])
        torch.testing.assert_close(output_residual, residual[::2])

    def test_mla_to_kda_gathers_hidden_states_and_residual(self):
        strategy = _RecordingStrategy()
        communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=True,
            previous_is_kda_layer=False,
        )
        hidden_states = torch.arange(4).view(2, 2)
        residual = hidden_states + 100

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
            output, output_residual = communicator.prepare_attn(
                hidden_states,
                residual,
                SimpleNamespace(),
            )

        self.assertEqual(strategy.gather_calls, 2)
        torch.testing.assert_close(output, hidden_states + 10)
        torch.testing.assert_close(output_residual, residual + 10)


if __name__ == "__main__":
    unittest.main()
