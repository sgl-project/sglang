"""Regressions for complete sequences and matching head shards under prefill CP."""

import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.kimi_linear import KimiDecoderLayer, KimiDeltaAttention, KimiMoE
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
_KIMI = "sglang.srt.models.kimi_linear."
_COMM = "sglang.srt.layers.communicator."


class _Identity(nn.Module):
    def forward(self, hidden_states, residual=None, *args, **kwargs):
        return hidden_states if residual is None else (hidden_states, residual)


def _communicator(*, linear=True, mlp_mode=ScatterMode.TP_ATTN_FULL, reduce_mlp=False):
    modes = LayerScatterModes(*([ScatterMode.TP_ATTN_FULL] * 5))
    modes.mlp_mode = mlp_mode
    with patch(_COMM + "get_moe_cp_size", return_value=get_parallel().attn_cp_size):
        return LayerCommunicator(
            modes,
            _Identity(),
            _Identity(),
            is_linear_attention=linear,
            reduce_mlp_output=reduce_mlp,
        )


class TestLinearAttnCPBoundary(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_prefill_cp=True, enable_linear_attn_cp=True, cp_strategy="zigzag"
        )
        override.install()
        self.addCleanup(override.restore)
        self.enterContext(
            get_parallel().override(
                tp_size=4,
                tp_rank=0,
                attn_tp_size=2,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_cp_size=2,
                attn_cp_rank=0,
            )
        )
        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        self.addCleanup(
            init_cp_strategy, enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag"
        )
        self.strategy = ZigzagCPStrategy(cp_size=2)
        lengths = [65, 113, 77]
        self.full = torch.arange(sum(lengths) * 2, dtype=torch.float32).reshape(-1, 2)
        self.batches, self.shards = [], []
        for rank in range(2):
            with get_parallel().override(attn_cp_size=2, attn_cp_rank=rank):
                meta = self.strategy.build_metadata(sum(lengths), lengths, lengths)
                pad_logical_token_to_physical(meta)
                batch = SimpleNamespace(
                    attn_cp_metadata=meta,
                    input_ids=torch.arange(sum(lengths)),
                    extend_seq_lens_cpu=lengths,
                    forward_mode=ForwardMode.EXTEND,
                )
                self.batches.append(batch)
                self.shards.append(self.strategy.shard_hidden_states(self.full, batch))

    def _group(self):
        group = Mock()

        def gather(output, local):
            rows = output.shape[0] // 2
            torch.cat([shard[:rows] for shard in self.shards], out=output)

        group.all_gather_into_tensor.side_effect = gather
        group.all_reduce.side_effect = lambda x: x + self.full
        return group

    def test_ragged_sequences_are_gathered_reduced_and_resharded(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                group, attn_group = self._group(), Mock()
                attn_group.all_reduce.side_effect = lambda x: x * 5
                with get_parallel().override(
                    attn_cp_rank=rank, attn_cp_group=group, attn_tp_group=attn_group
                ):
                    comm = _communicator()
                    comm.post_attention_layernorm = Mock(
                        side_effect=lambda x, residual: (x + 11, residual)
                    )
                    prepared, residual = comm.prepare_attn(
                        self.shards[rank], None, self.batches[rank]
                    )
                    output, residual = comm.prepare_mlp(
                        prepared * 2, residual, self.batches[rank]
                    )
                torch.testing.assert_close(prepared, self.full)
                expected = self.strategy.shard_hidden_states(
                    self.full * 15, self.batches[rank]
                )
                torch.testing.assert_close(
                    comm.post_attention_layernorm.call_args.args[0], expected
                )
                torch.testing.assert_close(output, expected + 11)
                self.assertIs(residual, self.shards[rank])
                group.all_reduce.assert_called_once()
                attn_group.all_reduce.assert_called_once()

    def test_decode_and_short_extend_still_reduce_head_shards(self):
        hidden = self.full[:3]
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            for cp_size, linear in ((1, True), (2, False), (2, True)):
                with self.subTest(mode=mode, cp_size=cp_size, linear=linear):
                    group, attn_group = Mock(), Mock()
                    group.all_reduce.side_effect = lambda x: x + 7
                    attn_group.all_reduce.side_effect = lambda x: x + 5
                    batch = SimpleNamespace(
                        input_ids=torch.arange(3),
                        forward_mode=mode,
                        extend_seq_lens_cpu=[3],
                        attn_cp_metadata=None,
                    )
                    with get_parallel().override(
                        tp_size=2 * cp_size,
                        attn_cp_size=cp_size,
                        attn_cp_group=group,
                        attn_tp_group=attn_group,
                    ):
                        comm = _communicator(linear=linear)
                        prepared, residual = comm.prepare_attn(hidden, None, batch)
                        output, _ = comm.prepare_mlp(prepared * 2, residual, batch)
                    torch.testing.assert_close(
                        output, hidden * 2 + 5 + (7 if cp_size > 1 and linear else 0)
                    )
                    group.all_gather_into_tensor.assert_not_called()
                    self.assertEqual(
                        group.all_reduce.call_count, int(cp_size > 1 and linear)
                    )

    def test_kimi_moe_reduces_weight_shards_for_matching_tokens(self):
        for enabled, rank in ((False, 0), (True, 0), (True, 1)):
            with self.subTest(enabled=enabled, rank=rank):
                batch = (
                    self.batches[rank]
                    if enabled
                    else SimpleNamespace(
                        forward_mode=ForwardMode.DECODE, attn_cp_metadata=None
                    )
                )
                hidden = self.shards[rank] if enabled else self.full[:3]
                if enabled:
                    rows = max(batch.attn_cp_metadata.per_rank_actual_token)
                    gathered = torch.cat(
                        [
                            torch.nn.functional.pad(x, (0, 0, 0, rows - len(x)))
                            for x in self.shards
                        ]
                    )
                else:
                    gathered = hidden
                moe = KimiMoE.__new__(KimiMoE)
                nn.Module.__init__(moe)
                moe.alt_stream, moe.num_shared_experts = None, None
                moe.gate = Mock(side_effect=lambda x: (x, None))
                moe.topk = Mock(return_value=None)
                moe.experts = Mock(side_effect=lambda x, topk: x * (rank + 1))
                layer = KimiDecoderLayer.__new__(KimiDecoderLayer)
                nn.Module.__init__(layer)
                layer.self_attn, layer.mlp = _Identity(), moe
                group = Mock()
                with (
                    get_context().override_server_args(enable_linear_attn_cp=enabled),
                    get_parallel().override(
                        tp_size=4 if enabled else 2,
                        attn_cp_size=2 if enabled else 1,
                        attn_cp_rank=rank,
                        attn_cp_group=group,
                    ),
                    patch(_COMM + "get_moe_cp_size", return_value=2 if enabled else 1),
                    patch(_COMM + "get_moe_cp_rank", return_value=rank),
                    patch(
                        _COMM + "moe_cp_all_gather_into_tensor",
                        side_effect=lambda out, x: out.copy_(gathered),
                    ),
                    patch(
                        _COMM + "attention_tensor_model_parallel_all_reduce",
                        side_effect=lambda x: x,
                    ),
                    patch(
                        _COMM + "tensor_model_parallel_all_reduce",
                        side_effect=lambda x: x + gathered * (2 - rank),
                    ) as reduce,
                ):
                    layer.layer_communicator = _communicator(
                        linear=False,
                        reduce_mlp=True,
                        mlp_mode=ScatterMode.MOE_FULL if enabled else ScatterMode.FULL,
                    )
                    output, residual = layer(None, hidden, batch, None, None)
                torch.testing.assert_close(moe.gate.call_args.args[0], gathered)
                torch.testing.assert_close(output, hidden * 3)
                self.assertIs(residual, hidden)
                reduce.assert_called_once()
                group.all_reduce.assert_not_called()


class TestKimiCPWeightGroups(CustomTestCase):
    def test_quantized_qkv_gates_and_registered_loaders_share_head_owners(self):
        config = SimpleNamespace(
            linear_attn_config={
                "num_heads": 16,
                "head_dim": 8,
                "short_conv_kernel_size": 4,
            }
        )

        def linear(*args, **kwargs):
            return SimpleNamespace(weight=nn.Parameter(torch.empty(8, 4)), bias=None)

        for on_attn_tp, rank, size in ((False, 6, 8), (True, 2, 4)):
            with self.subTest(on_attn_tp=on_attn_tp), ExitStack() as stack:
                stack.enter_context(
                    get_parallel().override(
                        tp_size=8, tp_rank=6, attn_tp_size=4, attn_tp_rank=2
                    )
                )
                mocks = {}
                for name in (
                    "QKVParallelLinear",
                    "ReplicatedLinear",
                    "ColumnParallelLinear",
                    "MergedColumnParallelLinear",
                    "RowParallelLinear",
                ):
                    mocks[name] = stack.enter_context(
                        patch(_KIMI + name, side_effect=linear)
                    )
                for name in ("FusedRMSNormGated", "RadixLinearAttention"):
                    stack.enter_context(patch(_KIMI + name))
                layer = KimiDeltaAttention(
                    0, 64, config, quant_config=Mock(), shard_on_attn_tp=on_attn_tp
                )
                for name in ("QKVParallelLinear", "MergedColumnParallelLinear"):
                    kwargs = mocks[name].call_args.kwargs
                    self.assertEqual(
                        (kwargs["tp_rank"], kwargs["tp_size"]), (rank, size)
                    )
                bias = torch.arange(128, dtype=torch.float32)
                layer.dt_bias.weight_loader(layer.dt_bias, bias)
                torch.testing.assert_close(layer.dt_bias, bias.chunk(size)[rank])
                decay = torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1)
                layer.A_log.weight_loader(layer.A_log, decay)
                torch.testing.assert_close(layer.A_log, decay.chunk(size, dim=2)[rank])

    def test_kimi_attention_reduction_belongs_to_communicator(self):
        config = SimpleNamespace(
            hidden_size=2,
            is_moe=False,
            is_kda_layer=lambda index: False,
            num_attention_heads=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            q_lora_rank=None,
            kv_lora_rank=2,
            intermediate_size=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        )
        for enabled, kda in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(enabled=enabled, kda=kda):
                config.is_kda_layer = lambda index: kda
                cp_group = Mock()
                cp_group.all_reduce.side_effect = lambda x: x + 7
                with (
                    get_context().override_server_args(enable_linear_attn_cp=enabled),
                    get_parallel().override(
                        tp_size=4 if enabled else 2,
                        attn_tp_size=2,
                        attn_dp_size=1,
                        attn_cp_size=2 if enabled else 1,
                        attn_cp_group=cp_group,
                        tp_rank=0,
                        attn_tp_rank=0,
                        attn_cp_rank=0,
                    ),
                    patch(_COMM + "get_moe_cp_size", return_value=2 if enabled else 1),
                    patch(
                        _KIMI + ("KimiDeltaAttention" if kda else "KimiMLAAttention"),
                        return_value=_Identity(),
                    ) as attention,
                    patch(_KIMI + "KimiMLP", return_value=_Identity()) as mlp,
                    patch(
                        _COMM + "attention_tensor_model_parallel_all_reduce",
                        side_effect=lambda x: x + 3,
                    ),
                ):
                    layer = KimiDecoderLayer(config, 0)
                    norm = Mock(side_effect=lambda x, residual: (x, residual))
                    layer.layer_communicator.post_attention_layernorm = norm
                    local = torch.tensor([[1.0, 2.0]])
                    batch = SimpleNamespace(
                        forward_mode=ForwardMode.DECODE, attn_cp_metadata=None
                    )
                    output, _ = layer.layer_communicator.prepare_mlp(
                        local, local, batch
                    )
                self.assertFalse(attention.call_args.kwargs["reduce_results"])
                self.assertFalse(mlp.call_args.kwargs["reduce_results"])
                expected = local + 3 + (7 if enabled and kda else 0)
                torch.testing.assert_close(norm.call_args.args[0], expected)
                torch.testing.assert_close(output, expected)
                self.assertEqual(cp_group.all_reduce.call_count, int(enabled and kda))


class TestLinearAttnCPResolution(CustomTestCase):
    def _resolve(self, architecture="Qwen3NextForCausalLM", **overrides):
        kimi = "Kimi" in architecture
        config = dict(
            architectures=[architecture],
            model_type="kimi_linear" if kimi else "qwen3_next",
            hidden_size=2048,
            num_hidden_layers=4,
            num_attention_heads=16,
            num_key_value_heads=16 if kimi else 2,
            torch_dtype="bfloat16",
            linear_num_key_heads=16,
            linear_num_value_heads=32,
        )
        if kimi:
            config.update(
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                linear_attn_config=dict(
                    num_heads=16,
                    head_dim=128,
                    short_conv_kernel_size=4,
                    kda_layers=[1, 2],
                    full_attn_layers=[3, 4],
                ),
            )
        with TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(config))
            args = dict(
                model_path=directory,
                tp_size=4,
                attn_cp_size=2,
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                skip_tokenizer_init=True,
            )
            args.update(overrides)
            server_args = ServerArgs(**args)
            server_args.resolve_once()
            return server_args

    def test_supported_models_enable_cp_and_disable_prefill_graphs(self):
        for architecture in ("Qwen3NextForCausalLM", "KimiLinearForCausalLM"):
            with self.subTest(architecture=architecture):
                args = self._resolve(architecture)
                self.assertTrue(resolution_result(args, "enable_linear_attn_cp"))
                graph = resolution_result(args, "cuda_graph_config")
                self.assertEqual(graph.prefill.backend, Backend.DISABLED)
        args = self._resolve(enable_prefill_cp=False, cp_strategy=None)
        self.assertFalse(resolution_result(args, "enable_linear_attn_cp"))

    def test_unsupported_execution_paths_fail_at_startup(self):
        for overrides, message in (
            ({"cp_strategy": "interleave"}, "cp-strategy"),
            ({"enable_mixed_chunk": True}, "enable-mixed-chunk"),
            ({"enable_dp_attention": True, "dp_size": 2}, "enable-dp-attention"),
            ({"moe_dp_size": 2}, "moe-dp-size"),
            ({"dcp_size": 2}, "dcp-size"),
            ({"tp_size": 64, "attn_cp_size": 8}, "linear_num_key_heads"),
            ({"architecture": "KimiK3LinearForCausalLM"}, "KimiK3"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._resolve(**overrides)


if __name__ == "__main__":
    unittest.main()
