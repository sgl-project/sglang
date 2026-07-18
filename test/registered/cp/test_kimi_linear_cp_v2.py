import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.cp.kimi_linear import KimiLinearCPV2LayerCommunicator
from sglang.srt.layers.cp.utils import CP_V2_DEFAULT_MODEL_CLASSES
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.models.kimi_linear import (
    KimiDecoderLayer,
    KimiDeltaAttention,
    KimiLinearForCausalLM,
)
from sglang.srt.runtime_context import get_parallel
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


class _SequencedFakeCPGroup:
    def __init__(self, *rank_tensor_sets):
        self.rank_tensor_sets = rank_tensor_sets
        self.call_index = 0

    def cp_all_gather_into_tensor_async(self, output, input_tensor, stream):
        del input_tensor, stream
        torch.cat(self.rank_tensor_sets[self.call_index], dim=0, out=output)
        self.call_index += 1


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

    def test_same_layout_and_inactive_cp_v2_are_noops(self):
        hidden_states = torch.arange(8).view(4, 2)
        residual = hidden_states + 100

        for is_kda_layer, previous_is_kda_layer, cp_v2_active in (
            (True, True, True),
            (False, False, True),
            (False, None, True),
            (True, False, False),
            (False, True, False),
        ):
            with self.subTest(
                is_kda_layer=is_kda_layer,
                previous_is_kda_layer=previous_is_kda_layer,
                cp_v2_active=cp_v2_active,
            ):
                strategy = _RecordingStrategy()
                communicator = KimiLinearCPV2LayerCommunicator(
                    is_kda_layer=is_kda_layer,
                    previous_is_kda_layer=previous_is_kda_layer,
                )
                with (
                    patch(
                        "sglang.srt.layers.cp.kimi_linear.is_cp_v2_active",
                        return_value=cp_v2_active,
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

                self.assertIs(output, hidden_states)
                self.assertIs(output_residual, residual)
                self.assertEqual(strategy.gather_calls, 0)
                self.assertEqual(strategy.shard_calls, 0)

    def test_zigzag_kda_mla_kda_round_trip_restores_token_order(self):
        cp_size = 4
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
        num_tokens = sum(extend_seq_lens)
        hidden_states = torch.arange(num_tokens * 2).view(num_tokens, 2)
        residual = hidden_states + 100
        strategy = ZigzagCPStrategy(cp_size=cp_size)
        shard_communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=False,
            previous_is_kda_layer=True,
        )
        gather_communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=True,
            previous_is_kda_layer=False,
        )

        forward_batches = []
        local_hidden_states = []
        local_residuals = []
        for rank in range(cp_size):
            with get_parallel().override(attn_cp_rank=rank):
                metadata = strategy.build_metadata(
                    num_tokens=num_tokens,
                    seqs_len=seq_lens,
                    extend_seqs_len=extend_seq_lens,
                )
            forward_batch = SimpleNamespace(attn_cp_metadata=metadata)
            forward_batches.append(forward_batch)
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
                local_hidden, local_residual = shard_communicator.prepare_attn(
                    hidden_states,
                    residual,
                    forward_batch,
                )
            local_hidden_states.append(local_hidden)
            local_residuals.append(local_residual)

        max_rank_len = forward_batches[0].attn_cp_metadata.max_rank_len[0]

        def _pad_rank_tensors(rank_tensors):
            return [
                torch.nn.functional.pad(
                    tensor,
                    [0, 0, 0, max_rank_len - tensor.shape[0]],
                )
                for tensor in rank_tensors
            ]

        padded_hidden_states = _pad_rank_tensors(local_hidden_states)
        padded_residuals = _pad_rank_tensors(local_residuals)

        for rank in range(cp_size):
            group = _SequencedFakeCPGroup(
                padded_hidden_states,
                padded_residuals,
            )
            with (
                get_parallel().override(attn_cp_group=group),
                patch(
                    "sglang.srt.layers.cp.kimi_linear.is_cp_v2_active",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.layers.cp.kimi_linear.get_cp_strategy",
                    return_value=strategy,
                ),
            ):
                gathered_hidden, gathered_residual = gather_communicator.prepare_attn(
                    local_hidden_states[rank],
                    local_residuals[rank],
                    forward_batches[rank],
                )

            torch.testing.assert_close(gathered_hidden, hidden_states)
            torch.testing.assert_close(gathered_residual, residual)


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
            patch(
                "sglang.srt.models.kimi_linear.KimiMLAAttention",
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

    def test_kda_backend_partitions_heads_over_global_tp(self):
        config = SimpleNamespace(
            linear_attn_config={
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
            v_head_dim=128,
            dtype=torch.bfloat16,
        )
        radix_linear_attention = MagicMock()

        with (
            get_parallel().override(
                tp_size=4,
                tp_rank=2,
                attn_tp_size=1,
                attn_tp_rank=0,
            ),
            patch(
                "sglang.srt.models.kimi_linear.MergedColumnParallelRepeatedLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.ColumnParallelBatchedLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.MergedColumnParallelLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.FusedRMSNormGated",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.RowParallelLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.RadixLinearAttention",
                return_value=radix_linear_attention,
            ) as radix_linear_attention_cls,
        ):
            KimiDeltaAttention(
                layer_idx=0,
                hidden_size=256,
                config=config,
            )

        radix_linear_attention_cls.assert_called_once()
        backend_args = radix_linear_attention_cls.call_args.kwargs
        self.assertEqual(backend_args["num_q_heads"], 8)
        self.assertEqual(backend_args["num_k_heads"], 8)
        self.assertEqual(backend_args["num_v_heads"], 8)


class TestKimiLinearCPV2Activation(CustomTestCase):
    def test_kimi_linear_uses_cp_v2_by_default(self):
        self.assertIn("KimiLinearForCausalLM", CP_V2_DEFAULT_MODEL_CLASSES)

    def test_causal_lm_exposes_input_embeddings(self):
        causal_lm = object.__new__(KimiLinearForCausalLM)
        embeddings = MagicMock()
        object.__setattr__(
            causal_lm,
            "model",
            SimpleNamespace(embed_tokens=embeddings),
        )

        self.assertIs(causal_lm.get_input_embeddings(), embeddings)


if __name__ == "__main__":
    unittest.main()
