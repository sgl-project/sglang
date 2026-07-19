import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
)
from sglang.srt.layers.cp.kimi_linear import KimiLinearCPV2LayerCommunicator
from sglang.srt.layers.cp.utils import CP_V2_DEFAULT_MODEL_CLASSES
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.models.kimi_linear import (
    KimiDecoderLayer,
    KimiDeltaAttention,
    KimiLinearForCausalLM,
    KimiLinearModel,
    _kda_head_sharded_weight_loader,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingStrategy:
    def __init__(self):
        self.gather_calls = 0
        self.gather_streams = []
        self.shard_calls = 0

    def gather_hidden_states(self, hidden_states, forward_batch, stream=None):
        del forward_batch
        self.gather_calls += 1
        self.gather_streams.append(stream)
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
    def test_mla_gathers_hidden_states_and_residual_before_mlp(self):
        strategy = _RecordingStrategy()
        communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=False,
            previous_is_kda_layer=True,
        )
        hidden_states = torch.arange(4).view(2, 2)
        residual = hidden_states + 100

        stream = MagicMock()
        with (
            patch(
                "sglang.srt.layers.cp.kimi_linear.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.cp.kimi_linear.get_cp_strategy",
                return_value=strategy,
            ),
            patch("torch.cuda.current_stream", return_value=stream),
        ):
            output, output_residual = communicator.prepare_mlp(
                hidden_states,
                residual,
                SimpleNamespace(),
            )

        self.assertEqual(strategy.gather_calls, 2)
        self.assertEqual(strategy.gather_streams, [stream, stream])
        torch.testing.assert_close(output, hidden_states + 10)
        torch.testing.assert_close(output_residual, residual + 10)

    def test_last_layer_shards_full_mlp_output_for_model_boundary_gather(self):
        strategy = _RecordingStrategy()
        communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=False,
            previous_is_kda_layer=True,
            is_last_layer=True,
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
            output, output_residual = communicator.postprocess_layer(
                hidden_states,
                residual,
                SimpleNamespace(),
            )

        self.assertEqual(strategy.shard_calls, 2)
        torch.testing.assert_close(output, hidden_states[::2])
        torch.testing.assert_close(output_residual, residual[::2])

    def test_first_kda_layer_gathers_model_entry_shard(self):
        strategy = _RecordingStrategy()
        communicator = KimiLinearCPV2LayerCommunicator(
            is_kda_layer=True,
            previous_is_kda_layer=None,
        )
        hidden_states = torch.arange(4).view(2, 2)

        stream = MagicMock()
        with (
            patch(
                "sglang.srt.layers.cp.kimi_linear.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.cp.kimi_linear.get_cp_strategy",
                return_value=strategy,
            ),
            patch("torch.cuda.current_stream", return_value=stream),
        ):
            output, residual = communicator.prepare_attn(
                hidden_states,
                None,
                SimpleNamespace(),
            )

        self.assertEqual(strategy.gather_calls, 1)
        self.assertEqual(strategy.gather_streams, [stream])
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

    def test_kda_after_mla_receives_full_mlp_output(self):
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

        self.assertEqual(strategy.gather_calls, 0)
        self.assertEqual(strategy.shard_calls, 0)
        self.assertIs(output, hidden_states)
        self.assertIs(output_residual, residual)

    def test_same_layout_and_inactive_cp_v2_are_noops(self):
        hidden_states = torch.arange(8).view(4, 2)
        residual = hidden_states + 100

        for is_kda_layer, previous_is_kda_layer, cp_v2_active in (
            (True, True, True),
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
                gathered_hidden, gathered_residual = shard_communicator.prepare_mlp(
                    local_hidden_states[rank],
                    local_residuals[rank],
                    forward_batches[rank],
                    stream=MagicMock(),
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
            num_hidden_layers=5,
            is_kda_layer=lambda layer_idx: layer_idx != 3,
        )
        communicator = MagicMock()
        hidden_states = torch.arange(8).view(2, 4)
        prepared_hidden_states = hidden_states + 10
        normalized_hidden_states = hidden_states + 20
        attention_output = hidden_states + 30
        gathered_attention_output = hidden_states + 35
        gathered_residual = hidden_states + 15
        post_norm_output = hidden_states + 40
        post_norm_residual = hidden_states + 50
        mlp_output = hidden_states + 60
        communicator.prepare_attn.return_value = (prepared_hidden_states, None)
        communicator.prepare_mlp.return_value = (
            gathered_attention_output,
            gathered_residual,
        )
        input_layernorm = MagicMock(return_value=normalized_hidden_states)
        self_attn = MagicMock(return_value=attention_output)
        post_attention_layernorm = MagicMock(
            return_value=(post_norm_output, post_norm_residual)
        )
        mlp = MagicMock(return_value=mlp_output)
        communicator.postprocess_layer.return_value = (
            mlp_output,
            post_norm_residual,
        )
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
            is_last_layer=False,
        )
        communicator.prepare_attn.assert_called_once_with(
            hidden_states,
            None,
            forward_batch,
        )
        input_layernorm.assert_called_once_with(prepared_hidden_states)
        communicator.prepare_mlp.assert_called_once_with(
            attention_output,
            prepared_hidden_states,
            forward_batch,
        )
        post_attention_layernorm.assert_called_once_with(
            gathered_attention_output,
            gathered_residual,
        )
        mlp.assert_called_once_with(post_norm_output)
        communicator.postprocess_layer.assert_called_once_with(
            mlp_output,
            post_norm_residual,
            forward_batch,
        )
        torch.testing.assert_close(output, mlp_output)
        torch.testing.assert_close(output_residual, post_norm_residual)

    def test_kda_backend_partitions_heads_over_cp(self):
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
                tp_size=8,
                tp_rank=6,
                attn_tp_size=2,
                attn_tp_rank=1,
                attn_cp_size=4,
                attn_cp_rank=2,
            ),
            patch(
                "sglang.srt.models.kimi_linear.QKVParallelLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.ReplicatedLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.ColumnParallelLinear",
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
                quant_config=MagicMock(),
            )

        radix_linear_attention_cls.assert_called_once()
        backend_args = radix_linear_attention_cls.call_args.kwargs
        self.assertEqual(backend_args["num_q_heads"], 8)
        self.assertEqual(backend_args["num_k_heads"], 8)
        self.assertEqual(backend_args["num_v_heads"], 8)

    def test_unfused_kda_projections_use_cp_rank_and_size(self):
        config = SimpleNamespace(
            linear_attn_config={
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
            v_head_dim=128,
            dtype=torch.bfloat16,
        )
        qkv_parallel_linear = MagicMock()
        column_parallel_linear = MagicMock()
        merged_column_parallel_linear = MagicMock()
        row_parallel_linear = MagicMock()

        with (
            get_parallel().override(
                tp_size=8,
                tp_rank=6,
                attn_tp_size=2,
                attn_tp_rank=1,
                attn_cp_size=4,
                attn_cp_rank=2,
            ),
            patch(
                "sglang.srt.models.kimi_linear.QKVParallelLinear",
                return_value=qkv_parallel_linear,
            ) as qkv_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.ReplicatedLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.ColumnParallelLinear",
                return_value=column_parallel_linear,
            ) as column_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.MergedColumnParallelLinear",
                return_value=merged_column_parallel_linear,
            ) as merged_column_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.FusedRMSNormGated",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.RowParallelLinear",
                return_value=row_parallel_linear,
            ) as row_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.RadixLinearAttention",
                return_value=MagicMock(),
            ),
        ):
            KimiDeltaAttention(
                layer_idx=0,
                hidden_size=256,
                config=config,
                quant_config=MagicMock(),
            )

        qkv_parallel_linear_cls.assert_called_once()
        projection_args = qkv_parallel_linear_cls.call_args.kwargs
        self.assertEqual(projection_args["tp_rank"], 2)
        self.assertEqual(projection_args["tp_size"], 4)

        self.assertEqual(column_parallel_linear_cls.call_count, 3)
        for call in column_parallel_linear_cls.call_args_list:
            self.assertEqual(call.kwargs["tp_rank"], 2)
            self.assertEqual(call.kwargs["tp_size"], 4)

        merged_column_parallel_linear_cls.assert_called_once()
        conv_args = merged_column_parallel_linear_cls.call_args.kwargs
        self.assertEqual(conv_args["tp_rank"], 2)
        self.assertEqual(conv_args["tp_size"], 4)

        row_parallel_linear_cls.assert_called_once()
        output_args = row_parallel_linear_cls.call_args.kwargs
        self.assertEqual(output_args["tp_rank"], 2)
        self.assertEqual(output_args["tp_size"], 4)
        self.assertFalse(output_args["reduce_results"])

    def test_kda_disables_global_tp_fusion_when_cp_owns_fewer_heads(self):
        config = SimpleNamespace(
            linear_attn_config={
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
            v_head_dim=128,
            dtype=torch.bfloat16,
        )

        with (
            get_parallel().override(
                tp_size=8,
                tp_rank=6,
                attn_tp_size=2,
                attn_tp_rank=1,
                attn_cp_size=4,
                attn_cp_rank=2,
            ),
            patch(
                "sglang.srt.models.kimi_linear.MergedColumnParallelRepeatedLinear",
                return_value=MagicMock(),
            ) as fused_projection_cls,
            patch(
                "sglang.srt.models.kimi_linear.QKVParallelLinear",
                return_value=MagicMock(),
            ) as qkv_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.ReplicatedLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.ColumnParallelLinear",
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
                return_value=MagicMock(),
            ),
        ):
            attention = KimiDeltaAttention(
                layer_idx=0,
                hidden_size=256,
                config=config,
            )

        self.assertFalse(attention.do_fuse_qkvbfg)
        fused_projection_cls.assert_not_called()
        qkv_parallel_linear_cls.assert_called_once()

    def test_kda_keeps_fusion_when_cp_matches_global_tp(self):
        config = SimpleNamespace(
            linear_attn_config={
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
            v_head_dim=128,
            dtype=torch.bfloat16,
        )

        with (
            get_parallel().override(
                tp_size=4,
                tp_rank=2,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=4,
                attn_cp_rank=2,
            ),
            patch(
                "sglang.srt.models.kimi_linear.MergedColumnParallelRepeatedLinear",
                return_value=MagicMock(),
            ) as fused_projection_cls,
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
            ) as row_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.RadixLinearAttention",
                return_value=MagicMock(),
            ),
        ):
            attention = KimiDeltaAttention(
                layer_idx=0,
                hidden_size=256,
                config=config,
            )

        self.assertTrue(attention.do_fuse_qkvbfg)
        fused_projection_cls.assert_called_once()
        self.assertEqual(attention.split_sizes, [3072, 8, 256])
        self.assertFalse(row_parallel_linear_cls.call_args.kwargs["reduce_results"])

    def test_kda_keeps_global_tp_projection_ownership_without_cp(self):
        config = SimpleNamespace(
            linear_attn_config={
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
            v_head_dim=128,
            dtype=torch.bfloat16,
        )

        with (
            get_parallel().override(
                tp_size=4,
                tp_rank=2,
                attn_tp_size=4,
                attn_tp_rank=2,
                attn_cp_size=1,
                attn_cp_rank=0,
            ),
            patch(
                "sglang.srt.models.kimi_linear.QKVParallelLinear",
                return_value=MagicMock(),
            ) as qkv_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.ReplicatedLinear",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.kimi_linear.ColumnParallelLinear",
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
            ) as row_parallel_linear_cls,
            patch(
                "sglang.srt.models.kimi_linear.RadixLinearAttention",
                return_value=MagicMock(),
            ),
        ):
            attention = KimiDeltaAttention(
                layer_idx=0,
                hidden_size=256,
                config=config,
                quant_config=MagicMock(),
            )

        projection_args = qkv_parallel_linear_cls.call_args.kwargs
        self.assertEqual(projection_args["tp_rank"], 2)
        self.assertEqual(projection_args["tp_size"], 4)
        output_args = row_parallel_linear_cls.call_args.kwargs
        self.assertTrue(output_args["reduce_results"])
        self.assertFalse(attention.kda_heads_use_cp)

    def test_kda_cache_shape_uses_cp_size(self):
        config = KimiLinearConfig(
            num_hidden_layers=2,
            linear_attn_config={
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
                "kda_layers": [1],
                "full_attn_layers": [2],
            },
        )

        with get_parallel().override(
            tp_size=8,
            attn_tp_size=2,
            attn_cp_size=4,
        ):
            shape = config.mamba2_cache_params.shape

        self.assertEqual(shape.temporal, (8, 128, 128))
        self.assertEqual(shape.conv, [(3, 3072)])
        self.assertEqual(shape.num_k_heads_per_tp, 8)

    def test_kda_state_weight_loader_uses_cp_rank(self):
        param = torch.nn.Parameter(torch.zeros(2))
        loaded_weight = torch.arange(16, dtype=param.dtype)
        loader = _kda_head_sharded_weight_loader(0)

        with get_parallel().override(
            tp_size=8,
            tp_rank=6,
            attn_tp_size=2,
            attn_tp_rank=1,
            attn_cp_size=4,
            attn_cp_rank=2,
        ):
            loader(param, loaded_weight)

        torch.testing.assert_close(param, torch.tensor([4.0, 5.0]))

    def test_kda_state_weight_loader_keeps_global_tp_without_cp(self):
        param = torch.nn.Parameter(torch.zeros(2))
        loaded_weight = torch.arange(8, dtype=param.dtype)
        loader = _kda_head_sharded_weight_loader(0)

        with get_parallel().override(
            tp_size=4,
            tp_rank=2,
            attn_tp_size=4,
            attn_tp_rank=2,
            attn_cp_size=1,
            attn_cp_rank=0,
        ):
            loader(param, loaded_weight)

        torch.testing.assert_close(param, torch.tensor([4.0, 5.0]))

    def test_kda_output_reduces_only_over_its_cp_head_group(self):
        for kda_heads_use_cp in (True, False):
            with self.subTest(kda_heads_use_cp=kda_heads_use_cp):
                attention = object.__new__(KimiDeltaAttention)
                torch.nn.Module.__init__(attention)
                attention.do_fuse_qkvbfg = True
                attention.head_dim = 2
                attention.kda_heads_use_cp = kda_heads_use_cp
                attention.forward_qkvbfg_fused = MagicMock(
                    return_value=(
                        torch.zeros(2, 12),
                        torch.zeros(2, 2),
                        torch.zeros(2, 4),
                        torch.zeros(2, 4),
                    )
                )
                attention.attn = MagicMock(return_value=torch.zeros(1, 2, 2, 2))
                attention.o_norm = MagicMock(return_value=torch.zeros(1, 2, 2, 2))
                partial_output = torch.arange(8, dtype=torch.float32).view(2, 4)
                attention.o_proj = MagicMock(return_value=(partial_output, None))
                cp_group = MagicMock()
                cp_group.all_reduce.return_value = partial_output + 10
                forward_batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_decode=lambda: True)
                )

                with get_parallel().override(attn_cp_group=cp_group):
                    output = attention(
                        hidden_states=torch.zeros(2, 4),
                        positions=torch.arange(2),
                        forward_batch=forward_batch,
                        zero_allocator=MagicMock(),
                    )

                if kda_heads_use_cp:
                    cp_group.all_reduce.assert_called_once_with(partial_output)
                    torch.testing.assert_close(output, partial_output + 10)
                else:
                    cp_group.all_reduce.assert_not_called()
                    torch.testing.assert_close(output, partial_output)


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

    def test_inner_model_accepts_cp_v2_input_embeds_keyword(self):
        parameters = inspect.signature(KimiLinearModel.forward).parameters

        self.assertIn("input_embeds", parameters)


class TestKimiLinearFlashInferMLACP(CustomTestCase):
    def test_cp_v2_skips_unused_full_batch_prefill_plan(self):
        backend = object.__new__(FlashInferMLAAttnBackend)
        backend.cp_prefill_metadata = MagicMock()
        backend.forward_metadata = MagicMock()
        backend.indices_updater_decode = MagicMock()
        backend.indices_updater_prefill = MagicMock()
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: False,
                is_target_verify=lambda: False,
            )
        )

        with patch(
            "sglang.srt.layers.attention.flashinfer_mla_backend.is_cp_v2_active",
            return_value=True,
        ):
            backend.init_forward_metadata(forward_batch)

        self.assertIsNone(backend.cp_prefill_metadata)
        self.assertIsNone(backend.forward_metadata)
        backend.indices_updater_decode.update.assert_not_called()
        backend.indices_updater_prefill.update.assert_not_called()

    def test_cp_wrapper_plan_uses_physical_token_page_size(self):
        backend = object.__new__(FlashInferMLAAttnBackend)
        backend.workspace_buffer = MagicMock()
        backend.page_size = 16
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((2, 32), dtype=torch.int32)
        )
        backend.indices_updater_prefill = SimpleNamespace(
            num_local_heads=4,
            kv_lora_rank=8,
            qk_rope_head_dim=4,
            scaling=0.5,
            q_data_type=torch.bfloat16,
            data_type=torch.bfloat16,
        )
        forward_batch = SimpleNamespace(req_pool_indices=torch.tensor([0, 1]))
        qo_indptr = torch.tensor([0, 2, 5], dtype=torch.int32)
        kv_lens = torch.tensor([3, 4], dtype=torch.int32)
        wrapper = MagicMock()
        indices_kernel = MagicMock()

        with (
            patch(
                "sglang.srt.layers.attention.flashinfer_mla_backend.BatchMLAPagedAttentionWrapper",
                return_value=wrapper,
                create=True,
            ),
            patch(
                "sglang.srt.layers.attention.flashinfer_mla_backend.create_flashinfer_kv_indices_triton",
                indices_kernel,
            ),
        ):
            backend._plan_cp_prefill_wrapper(
                forward_batch,
                qo_indptr,
                kv_lens,
                kv_lens_sum=7,
            )

        self.assertEqual(wrapper.plan.call_args.args[7], 1)

    def test_cp_v2_materializes_full_latent_and_dispatches_zigzag_attention(self):
        backend = object.__new__(FlashInferMLAAttnBackend)
        backend.device = torch.device("cpu")
        backend._get_cp_prefill_metadata = MagicMock(
            return_value=SimpleNamespace(wrappers=[MagicMock(), MagicMock()])
        )
        backend._run_cp_paged_attention = MagicMock()
        strategy = MagicMock()

        def run_attention(q_fused, forward_batch, device, attn_fn, **kwargs):
            del forward_batch, device, kwargs
            return torch.cat(
                [
                    attn_fn(q_fused[:2], None, None, None),
                    attn_fn(q_fused[2:], None, None, None),
                ]
            )

        strategy.run_attention.side_effect = run_attention
        q = torch.randn(5, 8)
        q_rope = torch.randn(5, 4)
        k = torch.randn(5, 8)
        v = torch.randn(5, 8)
        k_rope = torch.randn(5, 4)
        layer = SimpleNamespace(
            tp_q_head_num=2,
            v_head_dim=4,
            head_dim=6,
        )
        forward_batch = SimpleNamespace()
        backend._run_cp_paged_attention.side_effect = lambda wrapper, q_chunk, layer: (
            q_chunk[..., : layer.v_head_dim]
        )

        with (
            patch(
                "sglang.srt.layers.attention.flashinfer_mla_backend.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.flashinfer_mla_backend.get_cp_strategy",
                return_value=strategy,
            ),
        ):
            output = backend.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
            )

        strategy.materialize_full_mla_kv.assert_called_once_with(
            forward_batch,
            layer,
            k,
            k_rope,
        )
        strategy.run_attention.assert_called_once()
        self.assertEqual(backend._run_cp_paged_attention.call_count, 2)
        self.assertIs(
            backend._run_cp_paged_attention.call_args_list[0].args[0],
            backend._get_cp_prefill_metadata.return_value.wrappers[0],
        )
        self.assertIs(
            backend._run_cp_paged_attention.call_args_list[1].args[0],
            backend._get_cp_prefill_metadata.return_value.wrappers[1],
        )
        torch.testing.assert_close(output, q)


if __name__ == "__main__":
    unittest.main()
