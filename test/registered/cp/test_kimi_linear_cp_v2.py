import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.cp.kimi_linear import KimiLinearCPV2LayerCommunicator
from sglang.srt.models.kimi_linear import KimiDecoderLayer
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


class TestKimiDecoderLayerCPV2Wiring(CustomTestCase):
    def test_layer_prepares_cp_layout_before_input_norm(self):
        config = SimpleNamespace(
            hidden_size=4,
            num_attention_heads=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            q_lora_rank=None,
            kv_lora_rank=2,
            is_moe=False,
            intermediate_size=8,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            is_kda_layer=lambda layer_idx: layer_idx != 3,
        )
        communicator = MagicMock()
        hidden_states = torch.arange(8).view(2, 4)
        prepared_hidden_states = hidden_states + 10
        normalized_hidden_states = hidden_states + 20
        attention_output = hidden_states + 30
        post_norm_output = hidden_states + 40
        post_norm_residual = hidden_states + 50
        mlp_output = hidden_states + 60
        communicator.prepare_attn.return_value = (prepared_hidden_states, None)
        input_layernorm = MagicMock(return_value=normalized_hidden_states)
        self_attn = MagicMock(return_value=attention_output)
        post_attention_layernorm = MagicMock(
            return_value=(post_norm_output, post_norm_residual)
        )
        mlp = MagicMock(return_value=mlp_output)
        stream = MagicMock()
        forward_batch = SimpleNamespace()

        with (
            patch(
                "sglang.srt.models.kimi_linear.KimiLinearCPV2LayerCommunicator",
                return_value=communicator,
            ) as communicator_cls,
            patch(
                "sglang.srt.models.kimi_linear.KimiDeltaAttention",
                return_value=self_attn,
            ),
            patch("sglang.srt.models.kimi_linear.KimiMLP", return_value=mlp),
            patch(
                "sglang.srt.models.kimi_linear.RMSNorm",
                side_effect=[input_layernorm, post_attention_layernorm],
            ),
            patch("torch.cuda.current_stream", return_value=stream),
        ):
            layer = KimiDecoderLayer(config=config, layer_idx=3)
            output, output_residual = layer(
                positions=torch.arange(2),
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=None,
                zero_allocator=MagicMock(),
            )

        communicator_cls.assert_called_once_with(
            is_kda_layer=False,
            previous_is_kda_layer=True,
        )
        communicator.prepare_attn.assert_called_once_with(
            hidden_states,
            None,
            forward_batch,
            stream,
        )
        input_layernorm.assert_called_once_with(prepared_hidden_states)
        torch.testing.assert_close(output, mlp_output)
        torch.testing.assert_close(output_residual, post_norm_residual)


if __name__ == "__main__":
    unittest.main()
