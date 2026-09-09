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
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.layers.layer_boundary import SumGroup, ops
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.test.boundary_fixtures import build_stages, parallel_of, planning
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.kimi_linear import KimiDecoderLayer, KimiDeltaAttention, KimiMoE
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
_KIMI = "sglang.srt.models.kimi_linear."


class _Identity(nn.Module):
    def forward(self, hidden_states, residual=None, *args, **kwargs):
        return hidden_states if residual is None else (hidden_states, residual)


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
                moe_tp_size=4,
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
            with get_parallel().override(attn_cp_size=2, attn_cp_rank=rank, tp_rank=2 * rank):
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

    def test_ragged_sequences_are_gathered_reduced_and_resharded(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                cp_group, tp_group = Mock(), Mock()

                def gather(output, local):
                    rows = output.shape[0] // 2
                    torch.cat([shard[:rows] for shard in self.shards], out=output)

                cp_group.all_gather_into_tensor.side_effect = gather
                tp_group.all_reduce.side_effect = lambda x: x * 4
                with get_parallel().override(
                    tp_rank=2 * rank, attn_cp_rank=rank, attn_cp_group=cp_group, tp_group=tp_group
                ):
                    prepared = ops.cp_gather_in_token_order(
                        self.shards[rank], self.batches[rank]
                    )
                    output, residual = ops.cp_take_back_in_token_order(
                        prepared * 2,
                        self.shards[rank],
                        self.batches[rank],
                        group=SumGroup.TP,
                    )
                torch.testing.assert_close(prepared, self.full)
                expected = self.strategy.shard_hidden_states(
                    self.full * 8, self.batches[rank]
                )
                torch.testing.assert_close(output, expected)
                self.assertIs(residual, self.shards[rank])
                tp_group.all_reduce.assert_called_once()
                # Reduction must see aligned, complete sequences before sharding.
                torch.testing.assert_close(
                    tp_group.all_reduce.call_args.args[0], self.full * 2
                )
                cp_group.all_reduce.assert_not_called()

    def test_completed_attention_sum_is_not_repeated(self):
        group = Mock()
        with get_parallel().override(tp_rank=2, attn_cp_rank=1, tp_group=group):
            output, residual = ops.cp_take_back_in_token_order(
                self.full * 3, self.shards[1], self.batches[1], group=None
            )
        torch.testing.assert_close(output, self.shards[1] * 3)
        self.assertIs(residual, self.shards[1])
        group.all_reduce.assert_not_called()

    def test_kimi_moe_reduces_weight_shards_for_matching_tokens(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                batch = self.batches[rank]
                rows = batch.attn_cp_metadata.per_rank_actual_token
                max_rows = max(rows)
                gathered = torch.cat([
                    torch.nn.functional.pad(x[:n], (0, 0, 0, max_rows - n))
                    for x, n in zip(self.shards, rows)
                ])
                moe = KimiMoE.__new__(KimiMoE)
                nn.Module.__init__(moe)
                moe.alt_stream, moe.num_shared_experts = None, None
                moe.gate = Mock(side_effect=lambda x: (x, None))
                moe.topk = Mock(return_value=None)
                moe.experts = Mock(side_effect=lambda x, topk: x * (rank + 1))
                with (
                    patch.object(ops, "moe_cp_all_gather_into_tensor",
                        side_effect=lambda out, x: out.copy_(gathered)),
                    patch.object(ops, "get_moe_cp_rank", return_value=rank),
                ):
                    prepared = ops.moe_cp_gather(self.shards[rank], rows, 2)
                    partial = moe(prepared)
                    reduced = partial + gathered * (2 - rank)
                    output = ops.moe_cp_take_back(reduced, rows)
                torch.testing.assert_close(moe.gate.call_args.args[0], gathered)
                torch.testing.assert_close(output, self.shards[rank][:rows[rank]] * 3)


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
                for name in ("QKVParallelLinear", "MergedColumnParallelLinear", "RowParallelLinear"):
                    kwargs = mocks[name].call_args.kwargs
                    self.assertEqual(
                        kwargs["parallel_group"], "attn_tp" if on_attn_tp else "tp"
                    )
                bias = torch.arange(128, dtype=torch.float32)
                layer.dt_bias.weight_loader(layer.dt_bias, bias)
                torch.testing.assert_close(layer.dt_bias, bias.chunk(size)[rank])
                decay = torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1)
                layer.A_log.weight_loader(layer.A_log, decay)
                torch.testing.assert_close(layer.A_log, decay.chunk(size, dim=2)[rank])

    def test_kimi_attention_boundary_owns_the_matching_head_sum(self):
        config = SimpleNamespace(is_moe=False, num_hidden_layers=2)
        for enabled, kda in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(enabled=enabled, kda=kda):
                config.is_kda_layer = lambda index: kda
                with get_context().override_server_args(enable_linear_attn_cp=enabled):
                    attn, ffn = KimiDecoderLayer.stage_facts(config, 0)
                self.assertEqual(attn.in_token_order, enabled and kda)
                self.assertIs(attn.tp_group, SumGroup.TP if enabled and kda else SumGroup.ATTN_TP)
                parallel = parallel_of(attn_dp=1, attn_tp=2, attn_cp=2 if enabled else 1, enable_prefill_cp=enabled)
                with planning(parallel, dsa_cp=enabled):
                    attention, feed_forward = build_stages((attn, _Identity()), (ffn, _Identity()))
                ordinary = feed_forward.plan.paths[BatchVariant.ORDINARY].entry
                # Decode and short extends retain the full head-shard sum even
                # though they do not gather CP token shards.
                self.assertIs(ordinary.declared_sum, attn.tp_group)
                if enabled and kda:
                    cp = attention.plan.paths[BatchVariant.CONTEXT_PARALLEL].entry
                    self.assertIs(cp.input_move, ops.cp_gather_in_token_order)


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
