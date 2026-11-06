"""Regressions for complete sequences and matching head shards under prefill CP."""

import json
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.layers.layer_boundary import SumGroup, ops
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.srt.models.kimi_linear import KimiDecoderLayer, KimiDeltaAttention, KimiMoE
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.boundary_fixtures import build_stages, parallel_of, planning
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
_KIMI = "sglang.srt.models.kimi_linear."
_K3 = "sglang.srt.models.kimi_k3."


class _Identity(nn.Module):
    def forward(self, hidden_states, residual=None, *args, **kwargs):
        return hidden_states if residual is None else (hidden_states, residual)


class TestLinearAttnCPBoundary(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_prefill_cp=True,
            enable_cp_tp_group_sharing=True,
            cp_strategy="zigzag",
        )
        override.install()
        self.addCleanup(override.restore)
        self.enterContext(
            get_parallel().override(
                tp_size=2,
                tp_rank=0,
                moe_tp_size=2,
                attn_tp_size=1,
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
            with get_parallel().override(
                attn_cp_size=2, attn_cp_rank=rank, tp_rank=rank
            ):
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
                tp_group.all_reduce.side_effect = lambda x: x * 2
                with get_parallel().override(
                    tp_rank=rank,
                    attn_cp_rank=rank,
                    attn_cp_group=cp_group,
                    tp_group=tp_group,
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
                    self.full * 4, self.batches[rank]
                )
                torch.testing.assert_close(output, expected)
                self.assertIs(residual, self.shards[rank])
                tp_group.all_reduce.assert_called_once()
                # Reduction must see aligned, complete sequences before sharding.
                torch.testing.assert_close(
                    tp_group.all_reduce.call_args.args[0], self.full * 2
                )
                cp_group.all_reduce.assert_not_called()

    def test_k3_bank_stays_local_while_kda_reads_the_full_sequence(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.layer_boundary.residual.attn_bank import (
            AttnBank,
            AttnBankState,
        )

        config = SimpleNamespace(
            is_moe=False,
            attn_res_block_size=4,
            is_kda_layer=lambda _: True,
            num_experts=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            routed_expert_hidden_size=4,
        )
        for rank, sparse, kda in (
            (rank, sparse, kda)
            for rank in range(2)
            for sparse in (False, True)
            for kda in (False, True)
        ):
            with self.subTest(rank=rank, sparse=sparse, kda=kda):
                config.is_moe = sparse
                config.is_kda_layer = lambda _: kda
                local, batch = self.shards[rank], self.batches[rank]
                bank = AttnBank()
                bank.open(local, block_num=2)
                reads = []

                def aggregate(value, residual, *args, rows=None, **kwargs):
                    self.assertIsNone(rows)
                    self.assertEqual(value.shape[0], len(local))
                    self.assertEqual(bank.current.block_residual.shape[0], len(local))
                    reads.append(value.shape[0])
                    summed = value if residual is None else value + residual
                    return summed * 2, summed

                bank.current.forward = Mock(side_effect=aggregate)
                bank_ops = AttnBankState(bank, None, None, None, None).residual_ops()
                group = Mock(world_size=2, ranks=[0, 1])

                def gather(output, value):
                    scale = value.sum() / local.sum()
                    count = output.shape[0] // 2
                    output.copy_(torch.cat([x[:count] for x in self.shards]) * scale)

                group.all_gather_into_tensor.side_effect = gather
                group.all_reduce.side_effect = lambda x: x * 2
                parallel = parallel_of(
                    attn_dp=1,
                    attn_tp=1,
                    attn_cp=2,
                    enable_prefill_cp=True,
                    enable_cp_tp_group_sharing=True,
                    tp_group=group,
                    attn_cp_group=group,
                    tp_rank=rank,
                    attn_cp_rank=rank,
                )
                with (
                    get_parallel().override(
                        tp_rank=rank,
                        attn_cp_rank=rank,
                        tp_group=group,
                        attn_cp_group=group,
                    ),
                    planning(parallel, dsa_cp=True),
                    patch_communicator("dsa_use_prefill_cp", lambda batch: True),
                ):
                    attn, ffn = kimi_k3._stage_facts(
                        config, 0, sp_collective=False, bank=bank_ops
                    )
                    attention, feed_forward = build_stages((attn, None), (ffn, None))
                    residual_batch.start(batch)
                    operand = attention.prepare(local, batch)
                    torch.testing.assert_close(
                        operand, (self.full if kda else local) * 2
                    )
                    self.assertIs(batch.residual_stream.residual, local)
                    output = attention.finish(operand * (1.5 if kda else 3), batch)
                    ffn_operand = feed_forward.prepare(output, batch)
                    torch.testing.assert_close(
                        batch.residual_stream.residual, local * 7
                    )
                    if sparse:
                        torch.testing.assert_close(ffn_operand, local * 14)
                        # Latent MoE returns its completed local result including
                        # the prefix residual; the boundary must not add it twice.
                        output = feed_forward.finish(
                            ffn_operand / 2 + batch.residual_stream.residual, batch
                        )
                        self.assertIsNone(batch.residual_stream.transport)
                        output = residual_batch.final_norm(
                            output,
                            batch,
                            lambda x, residual=None: (
                                x if residual is None else (x + residual, x + residual)
                            ),
                        )
                        torch.testing.assert_close(output, local * 14)
                    else:
                        residual_batch.clear(batch)
                self.assertEqual(reads, [len(local), len(local)])
                self.assertEqual(bank.current.block_residual.shape[0], len(local))
                self.assertEqual(group.all_reduce.call_count, int(kda))

    def test_k3_latent_decoder_keeps_plain_residuals_on_local_rows(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        config = SimpleNamespace(
            is_moe=True,
            num_experts=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            routed_expert_hidden_size=4,
            attn_res_block_size=None,
        )

        def norm(value, residual=None):
            if residual is None:
                return value
            return value + residual, value + residual

        for rank, kda in ((0, False), (1, False), (0, True), (1, True)):
            with self.subTest(rank=rank, kda=kda):
                config.is_kda_layer = lambda _: kda
                local, batch = self.shards[rank], self.batches[rank]
                group = Mock(world_size=2, ranks=[0, 1])

                def gather(output, value):
                    rows = output.shape[0] // 2
                    output.copy_(torch.cat([x[:rows] for x in self.shards]))

                group.all_gather_into_tensor.side_effect = gather
                group.all_reduce.side_effect = lambda x: x * 2
                parallel = parallel_of(
                    attn_dp=1,
                    attn_tp=1,
                    attn_cp=2,
                    enable_prefill_cp=True,
                    enable_cp_tp_group_sharing=True,
                    tp_group=group,
                    attn_cp_group=group,
                    tp_rank=rank,
                    attn_cp_rank=rank,
                )
                with (
                    get_parallel().override(
                        tp_rank=rank,
                        attn_cp_rank=rank,
                        tp_group=group,
                        attn_cp_group=group,
                    ),
                    planning(parallel, dsa_cp=True),
                    patch_communicator("dsa_use_prefill_cp", lambda batch: True),
                ):
                    attn, ffn = kimi_k3._stage_facts(config, 0, sp_collective=False)
                    attention, feed_forward = build_stages((attn, norm), (ffn, norm))
                    layer = kimi_k3.KimiK3DecoderLayer.__new__(
                        kimi_k3.KimiK3DecoderLayer
                    )
                    nn.Module.__init__(layer)
                    layer.attn_boundary, layer.ffn_boundary = attention, feed_forward
                    layer._is_moe_layer = True
                    layer._ffn_writes_stream = False
                    layer._run_self_attn = Mock(
                        side_effect=lambda value, *args: value * (2 if kda else 4)
                    )
                    layer.mlp = Mock(
                        use_latent_moe=True,
                        side_effect=lambda value, **kwargs: value * 3,
                    )
                    residual_batch.start(batch)
                    output = layer(torch.arange(len(local)), local, batch, None)
                    output = residual_batch.final_norm(output, batch, norm)

                # Attention contributes 4x, then MoE contributes 3 * (1 + 4)x.
                torch.testing.assert_close(output, local * 20)
                torch.testing.assert_close(layer.mlp.call_args.args[0], local * 5)
                self.assertIs(layer.mlp.call_args.kwargs["forward_batch"], batch)
                self.assertIsNone(layer.mlp.call_args.kwargs["prefix_sum"])
                self.assertIsNone(ffn.row_transport)
                self.assertEqual(group.all_reduce.call_count, int(kda))

    def test_completed_attention_sum_is_not_repeated(self):
        group = Mock()
        with get_parallel().override(tp_rank=1, attn_cp_rank=1, tp_group=group):
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
                gathered = torch.cat(
                    [
                        torch.nn.functional.pad(x[:n], (0, 0, 0, max_rows - n))
                        for x, n in zip(self.shards, rows)
                    ]
                )
                moe = KimiMoE.__new__(KimiMoE)
                nn.Module.__init__(moe)
                moe.alt_stream, moe.num_shared_experts = None, None
                moe.gate = Mock(side_effect=lambda x: (x, None))
                moe.topk = Mock(return_value=None)
                moe.experts = Mock(side_effect=lambda x, topk: x * (rank + 1))
                with (
                    patch.object(
                        ops,
                        "moe_cp_all_gather_into_tensor",
                        side_effect=lambda out, x: out.copy_(gathered),
                    ),
                    patch.object(ops, "get_moe_cp_rank", return_value=rank),
                ):
                    prepared = ops.moe_cp_gather(self.shards[rank], rows, 2)
                    partial = moe(prepared)
                    reduced = partial + gathered * (2 - rank)
                    output = ops.moe_cp_take_back(reduced, rows)
                torch.testing.assert_close(moe.gate.call_args.args[0], gathered)
                torch.testing.assert_close(output, self.shards[rank][: rows[rank]] * 3)


class TestKimiCPWeightGroups(CustomTestCase):
    def test_k3_latent_moe_boundary_keeps_local_cp_rows(self):
        from sglang.srt.layers.layer_boundary import TokenAxis

        config = SimpleNamespace(
            is_moe=True,
            num_experts=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            routed_expert_hidden_size=4,
            attn_res_block_size=4,
            is_kda_layer=lambda _: True,
        )
        parallel = parallel_of(
            attn_dp=1,
            attn_tp=1,
            attn_cp=2,
            enable_prefill_cp=True,
            enable_cp_tp_group_sharing=True,
        )
        for kda in (False, True):
            config.is_kda_layer = lambda _: kda
            with (
                get_context().override_server_args(enable_cp_tp_group_sharing=True),
                get_parallel().override(
                    tp_size=2,
                    tp_rank=0,
                    moe_tp_size=2,
                    attn_tp_size=1,
                    attn_tp_rank=0,
                    attn_cp_size=2,
                    attn_cp_rank=0,
                ),
                planning(parallel, dsa_cp=True),
            ):
                attn, ffn = kimi_k3._stage_facts(config, 0, sp_collective=False)
                self.assertIsNone(ffn.row_transport)
                _, feed_forward = build_stages((attn, None), (ffn, None))
            for variant, expected in (
                (BatchVariant.ORDINARY, frozenset()),
                (BatchVariant.CONTEXT_PARALLEL, frozenset({TokenAxis.ATTN_CP})),
            ):
                path = feed_forward.plan.paths[variant]
                self.assertEqual(path.entry.input_rows.sharded, expected)
                self.assertEqual(path.output.layout.sharded, expected)
                self.assertIsNone(path.entry.input_move)
                self.assertIsNone(path.output.group)

    def test_k3_kda_projections_loaders_and_state_use_matching_tp_shards(self):
        config = KimiLinearConfig(
            v_head_dim=8,
            linear_attn_config=dict(
                num_heads=16,
                head_dim=8,
                short_conv_kernel_size=4,
                kda_layers=[1, 2],
                full_attn_layers=[3, 4],
            ),
        )

        def linear(*args, **kwargs):
            return SimpleNamespace(weight=nn.Parameter(torch.empty(8, 4)), bias=None)

        for enabled, cp_size in ((False, 1), (True, 8)):
            for full_rank_gate in (False, True):
                with (
                    self.subTest(
                        enabled=enabled, cp_size=cp_size, full_rank_gate=full_rank_gate
                    ),
                    ExitStack() as stack,
                ):
                    stack.enter_context(
                        get_context().override_server_args(
                            enable_cp_tp_group_sharing=enabled
                        )
                    )
                    stack.enter_context(
                        get_parallel().override(
                            tp_size=8,
                            moe_tp_size=8,
                            tp_rank=6,
                            attn_tp_size=8 // cp_size,
                            attn_tp_rank=6 % (8 // cp_size),
                            attn_cp_size=cp_size,
                            attn_cp_rank=6 // (8 // cp_size),
                        )
                    )
                    rank = 6 if enabled else 6 % (8 // cp_size)
                    size = 8 if enabled else 8 // cp_size
                    config.linear_attn_config["use_full_rank_gate"] = full_rank_gate
                    mocks = {}
                    for name in (
                        "QKVParallelLinear",
                        "ReplicatedLinear",
                        "ColumnParallelLinear",
                        "MergedColumnParallelLinear",
                        "RowParallelLinear",
                    ):
                        mocks[name] = stack.enter_context(
                            patch(_K3 + name, side_effect=linear)
                        )
                    stack.enter_context(patch(_K3 + "FusedRMSNormGated"))
                    attn = stack.enter_context(patch(_K3 + "RadixLinearAttention"))
                    layer = KimiK3DeltaAttention(0, 64, config, quant_config=Mock())
                    self.assertEqual(layer.local_num_heads, 16 // size)
                    for name in (
                        "QKVParallelLinear",
                        "ColumnParallelLinear",
                        "MergedColumnParallelLinear",
                        "RowParallelLinear",
                    ):
                        for call in mocks[name].call_args_list:
                            self.assertEqual(
                                call.kwargs["parallel_group"],
                                "tp" if enabled else "attn_tp",
                            )
                    output_args = mocks["RowParallelLinear"].call_args.kwargs
                    self.assertFalse(output_args["reduce_results"])
                    self.assertEqual(
                        output_args["parallel_group"], "tp" if enabled else "attn_tp"
                    )
                    for name in ("num_q_heads", "num_k_heads", "num_v_heads"):
                        self.assertEqual(attn.call_args.kwargs[name], 16 // size)

                    bias = torch.arange(128, dtype=torch.float32)
                    layer.dt_bias.weight_loader(layer.dt_bias, bias)
                    torch.testing.assert_close(layer.dt_bias, bias.chunk(size)[rank])
                    decay = torch.arange(16, dtype=torch.float32)
                    for weight in (decay, decay.view(1, 1, 16, 1)):
                        layer.A_log.weight_loader(layer.A_log, weight)
                        torch.testing.assert_close(
                            layer.A_log.flatten(), decay.chunk(size)[rank]
                        )

                    # The multimodal wrapper's nested text config must size
                    # recurrent and convolution state with the same head owner.
                    nested = KimiK3Config(text_config=config)
                    for text_config in (config, nested.text_config):
                        state = text_config.mamba2_cache_params.shape
                        self.assertEqual(state.temporal, (16 // size, 8, 8))
                        self.assertEqual(state.conv, [(3, 3 * 128 // size)])
                        self.assertEqual(
                            state.num_k_heads_per_tp, layer.local_num_heads
                        )

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
                for name in (
                    "QKVParallelLinear",
                    "MergedColumnParallelLinear",
                    "RowParallelLinear",
                ):
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
        for enabled, kda in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(enabled=enabled, kda=kda):
                config.is_kda_layer = lambda index: kda
                with get_context().override_server_args(
                    enable_cp_tp_group_sharing=enabled
                ):
                    attn, ffn = KimiDecoderLayer.stage_facts(config, 0)
                self.assertEqual(attn.in_token_order, kda)
                self.assertIs(
                    attn.tp_group, SumGroup.TP if enabled and kda else SumGroup.ATTN_TP
                )
                parallel = parallel_of(
                    attn_dp=1,
                    attn_tp=1 if enabled else 2,
                    attn_cp=2 if enabled else 1,
                    enable_prefill_cp=enabled,
                    enable_cp_tp_group_sharing=enabled,
                )
                with planning(parallel, dsa_cp=enabled):
                    attention, feed_forward = build_stages(
                        (attn, _Identity()), (ffn, _Identity())
                    )
                ordinary = feed_forward.plan.paths[BatchVariant.ORDINARY].entry
                # Decode and short extends retain the full head-shard sum even
                # though they do not gather CP token shards.
                self.assertIs(
                    ordinary.declared_sum,
                    (SumGroup.ATTN_CP if kda else None)
                    if enabled
                    else SumGroup.ATTN_TP,
                )
                if enabled and kda:
                    cp = attention.plan.paths[BatchVariant.CONTEXT_PARALLEL].entry
                    self.assertIs(cp.input_move, ops.cp_gather_in_token_order)


class TestKimiK3CPMoE(CustomTestCase):
    def _make_cp_moe(self, rank, *, shared, overlap=False):
        hidden = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10
        down = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 20
        up = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 30
        router = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 40
        routed, logits = hidden @ down, hidden @ router
        group = Mock(world_size=2)
        batch = SimpleNamespace(
            attn_cp_metadata=object(),
            num_token_non_padded=torch.tensor(3),
            global_num_token_non_padded_cpu=3,
        )
        state = SimpleNamespace(
            hidden=hidden,
            local=hidden.chunk(2)[rank],
            routed=routed,
            up=up,
            batch=batch,
            original_count=batch.num_token_non_padded,
            group=group,
            rank=rank,
            events=[],
            current_stream=Mock(),
            on_shared_stream=False,
        )
        gathered = {3: routed, 4: logits, 6: hidden}

        def gather(output, value):
            full = gathered[value.shape[1]]
            torch.testing.assert_close(value, full.chunk(2)[rank])
            output.copy_(full)

        def scatter(output, value):
            shared_output = value.shape[1] == 6
            full = hidden if shared_output else routed
            if shared_output and overlap:
                self.assertIn("wait_shared", state.events)
            state.events.append("scatter_shared" if shared_output else "scatter_latent")
            torch.testing.assert_close(value, full * (rank + 1))
            output.copy_((full * 3).chunk(2)[rank])

        def topk(values, scores):
            torch.testing.assert_close(values, routed)
            torch.testing.assert_close(scores, logits)
            self.assertEqual(batch.num_token_non_padded.item(), 4)
            self.assertEqual(batch.global_num_token_non_padded_cpu, 4)
            return None

        def experts(value, _):
            state.events.append("experts")
            self.assertEqual(batch.num_token_non_padded.item(), 4)
            return value * (rank + 1)

        def shared_experts(value):
            self.assertEqual(state.on_shared_stream, overlap)
            state.events.append("shared")
            torch.testing.assert_close(value, hidden)
            return value * (rank + 1)

        group.all_gather_into_tensor.side_effect = gather
        group.reduce_scatter_tensor.side_effect = scatter
        moe = kimi_k3.KimiK3MoE.__new__(kimi_k3.KimiK3MoE)
        nn.Module.__init__(moe)
        moe.use_latent_moe = True
        moe.tp_size = 2
        moe._shared_experts_tp1 = False
        moe._shared_experts_tp_comm = False
        moe._shared_experts_tp_group = None
        moe._sbo_shared_overlap = False
        moe._use_mega_moe = False
        moe.alt_stream = Mock() if overlap else None
        # CP must bypass a front that would otherwise be eligible.
        moe._eligible_for_fused_front = True
        moe._forward_fused = Mock(side_effect=AssertionError("CP used the fused path"))
        moe._ep_front = Mock(side_effect=AssertionError("CP routed local rows"))
        moe._ep_front_overlap = Mock(side_effect=AssertionError("CP routed local rows"))
        moe._can_overlap_shared_experts_npu = Mock(
            side_effect=AssertionError("CP entered the DeepEP overlap path")
        )
        moe._reduce_latent = Mock(side_effect=AssertionError("CP used all-reduce"))
        moe._reduce_shared = Mock(side_effect=AssertionError("CP reduced shared twice"))
        moe.gate = Mock(side_effect=lambda x: x @ router)
        moe.routed_expert_down_proj = Mock(side_effect=lambda x: (x @ down, None))
        moe.routed_expert_up_proj = Mock(side_effect=lambda x: (x @ up, None))
        moe._latent_norm = Mock(side_effect=lambda x: x.square())
        moe.shared_experts = Mock(side_effect=shared_experts) if shared else None
        moe.topk = Mock(side_effect=topk)
        moe.experts = Mock(side_effect=experts)
        state.moe = moe
        return state

    def _cp_context(self, state):
        stack = ExitStack()
        stack.enter_context(
            get_context().override_server_args(enable_cp_tp_group_sharing=True)
        )
        stack.enter_context(
            get_parallel().override(
                tp_size=2,
                tp_rank=state.rank,
                moe_tp_size=2,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=2,
                attn_cp_rank=state.rank,
                attn_cp_group=state.group,
            )
        )
        stack.enter_context(patch(_K3 + "is_cp_active", return_value=True))
        stack.enter_context(patch(_K3 + "_is_npu", False))
        stack.enter_context(
            patch(
                _K3 + "use_symmetric_memory", side_effect=lambda *a, **kw: nullcontext()
            )
        )
        if state.moe.alt_stream is not None:
            alt_stream = state.moe.alt_stream
            shared_event = object()
            alt_stream.record_event.return_value = shared_event

            def wait_shared(event):
                self.assertIs(event, shared_event)
                state.events.append("wait_shared")

            state.current_stream.wait_event.side_effect = wait_shared

            class SharedStream:
                def __enter__(self):
                    state.on_shared_stream = True

                def __exit__(self, *args):
                    state.on_shared_stream = False

            def shared_stream(stream):
                self.assertIs(stream, alt_stream)
                return SharedStream()

            stack.enter_context(
                patch.object(
                    torch.cuda, "current_stream", return_value=state.current_stream
                )
            )
            stack.enter_context(
                patch.object(torch.cuda, "stream", side_effect=shared_stream)
            )
            stack.enter_context(
                patch.object(torch.Tensor, "record_stream", return_value=None)
            )
        return stack

    def test_latent_projections_stay_local_and_reduce_before_normalization(self):
        for rank in (0, 1):
            for shared, overlap in ((False, False), (True, False), (True, True)):
                for with_prefix in (False, True):
                    with self.subTest(
                        rank=rank, shared=shared, overlap=overlap, prefix=with_prefix
                    ):
                        state = self._make_cp_moe(rank, shared=shared, overlap=overlap)
                        moe = state.moe
                        prefix = (
                            torch.full_like(state.local, 0.5) if with_prefix else None
                        )
                        with self._cp_context(state):
                            output = moe(
                                state.local,
                                forward_batch=state.batch,
                                prefix_sum=prefix,
                            )
                        latent = (state.routed * 3).chunk(2)[rank]
                        expected = latent.square() @ state.up
                        if prefix is not None:
                            expected += prefix
                        if shared:
                            expected += state.local * 3
                        torch.testing.assert_close(output, expected)
                        torch.testing.assert_close(
                            moe.gate.call_args.args[0], state.local
                        )
                        torch.testing.assert_close(
                            moe.routed_expert_down_proj.call_args.args[0], state.local
                        )
                        torch.testing.assert_close(
                            moe._latent_norm.call_args.args[0], latent
                        )
                        torch.testing.assert_close(
                            moe.routed_expert_up_proj.call_args.args[0], latent.square()
                        )
                        self.assertIs(
                            state.batch.num_token_non_padded, state.original_count
                        )
                        self.assertEqual(state.batch.global_num_token_non_padded_cpu, 3)
                        self.assertEqual(
                            state.group.reduce_scatter_tensor.call_count,
                            2 if shared else 1,
                        )
                        state.group.all_reduce.assert_not_called()
                        moe._forward_fused.assert_not_called()
                        moe._ep_front.assert_not_called()
                        moe._ep_front_overlap.assert_not_called()
                        if overlap:
                            moe.alt_stream.wait_stream.assert_called_once_with(
                                state.current_stream
                            )
                            self.assertLess(
                                state.events.index("shared"),
                                state.events.index("experts"),
                            )
                            self.assertLess(
                                state.events.index("wait_shared"),
                                state.events.index("scatter_shared"),
                            )

    def test_cp_restores_padding_metadata_when_experts_fail(self):
        state = self._make_cp_moe(0, shared=False)
        state.moe.experts.side_effect = RuntimeError("expert failure")
        with (
            self._cp_context(state),
            self.assertRaisesRegex(RuntimeError, "expert failure"),
        ):
            state.moe(state.local, forward_batch=state.batch)
        self.assertIs(state.batch.num_token_non_padded, state.original_count)
        self.assertEqual(state.batch.global_num_token_non_padded_cpu, 3)
        state.group.reduce_scatter_tensor.assert_not_called()

    def test_inactive_cp_preserves_fused_and_unfused_dispatch(self):
        hidden = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        prefix = torch.full_like(hidden, 0.5)
        batch = SimpleNamespace(attn_cp_metadata=object())
        for eligible in (False, True):
            with self.subTest(fused=eligible):
                moe = kimi_k3.KimiK3MoE.__new__(kimi_k3.KimiK3MoE)
                nn.Module.__init__(moe)
                moe.use_latent_moe = True
                moe._eligible_for_fused_front = eligible
                moe._forward_fused = Mock(return_value=hidden + 1)
                moe._forward_unfused = Mock(return_value=hidden + 2)
                with patch(_K3 + "_shards_rows_over_cp", return_value=False):
                    output = moe(hidden, forward_batch=batch, prefix_sum=prefix)
                selected = moe._forward_fused if eligible else moe._forward_unfused
                unselected = moe._forward_unfused if eligible else moe._forward_fused
                selected.assert_called_once()
                unselected.assert_not_called()
                self.assertIs(selected.call_args.kwargs["prefix_sum"], prefix)
                torch.testing.assert_close(output, hidden + (1 if eligible else 2))


class TestCPModelBoundaries(CustomTestCase):
    def test_models_shard_once_before_layers_and_gather_after_final_norm(self):
        from sglang.srt.models import kimi_linear

        full = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        positions = torch.arange(8)
        local, local_positions = full[::2], positions[::2]
        for module, model_cls in (
            (kimi_linear, kimi_linear.KimiLinearModel),
            (kimi_k3, kimi_k3.KimiK3LinearModel),
        ):
            for active in (False, True):
                with self.subTest(model=model_cls.__name__, active=active):
                    layer = Mock(side_effect=lambda **kw: kw["hidden_states"] * 3)
                    owner = SimpleNamespace(
                        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
                        layers=[layer],
                        start_layer=0,
                        end_layer=1,
                        norm=object(),
                        _final_read=object(),
                        attn_bank=None,
                        dspark_layers_to_capture=None,
                        packs_aux_hidden_states=False,
                    )
                    batch = SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND, attn_cp_metadata=object()
                    )
                    recorder = Mock()
                    recorder.with_current_layer.side_effect = lambda _: nullcontext()
                    with (
                        get_context().override_server_args(
                            enable_cp_tp_group_sharing=True
                        ),
                        get_parallel().override(pp_group=owner.pp_group),
                        patch.object(module, "is_cp_active", return_value=active),
                        patch.object(
                            module,
                            "cp_split_before_forward",
                            return_value=(local, local_positions),
                        ) as split,
                        patch.object(
                            module, "cp_gather_after_forward", return_value=full * 3
                        ) as gather,
                        patch.object(
                            module,
                            "get_global_expert_distribution_recorder",
                            return_value=recorder,
                        ),
                        patch.object(module.residual_batch, "start"),
                        patch.object(
                            module.residual_batch,
                            "complete_output",
                            side_effect=lambda x, *a, **kw: x,
                        ),
                        patch.object(
                            module.residual_batch,
                            "final_norm",
                            side_effect=lambda x, *a, **kw: x,
                        ) as norm,
                    ):
                        output = model_cls.forward(
                            owner, torch.arange(8), positions, batch, full
                        )
                    expected_input = local if active else full
                    torch.testing.assert_close(
                        layer.call_args.kwargs["hidden_states"], expected_input
                    )
                    torch.testing.assert_close(
                        norm.call_args.args[0], expected_input * 3
                    )
                    torch.testing.assert_close(output, full * 3)
                    if active:
                        split.assert_called_once_with(full, positions, batch)
                        gather.assert_called_once_with(norm.call_args.args[0], batch)
                    else:
                        split.assert_not_called()
                        gather.assert_not_called()


class TestLinearAttnCPResolution(CustomTestCase):
    def _resolve(
        self,
        architecture="KimiLinearForCausalLM",
        *,
        config_overrides=None,
        **overrides,
    ):
        from sglang.srt.runtime_context import override_platform

        kimi = "Kimi" in architecture
        kimi_k3 = architecture in (
            "KimiK3LinearForCausalLM",
            "KimiK3ForConditionalGeneration",
        )
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
            del config["linear_num_key_heads"], config["linear_num_value_heads"]
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
        config.update(config_overrides or {})
        if architecture == "KimiK3ForConditionalGeneration":
            config["architectures"] = ["KimiK3LinearForCausalLM"]
            config = dict(
                architectures=[architecture],
                model_type="kimi_k3",
                text_config=config,
                torch_dtype="bfloat16",
            )
        with TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(config))
            args = dict(
                model_path=directory,
                # Argument resolution also runs on accelerator-less CPU CI.
                device="cuda",
                tp_size=4,
                attn_cp_size=4,
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                skip_tokenizer_init=True,
            )
            if kimi_k3:
                args["attention_backend"] = "fa3"
            args.update(overrides)
            server_args = ServerArgs(**args)
            with override_platform(is_cuda=True):
                server_args.resolve_once()
            return server_args

    def test_supported_models_enable_cp_and_disable_prefill_graphs(self):
        for architecture in (
            "KimiLinearForCausalLM",
            "KimiK3LinearForCausalLM",
            "KimiK3ForConditionalGeneration",
        ):
            with self.subTest(architecture=architecture):
                args = self._resolve(architecture)
                self.assertTrue(resolution_result(args, "enable_cp_tp_group_sharing"))
                graph = resolution_result(args, "cuda_graph_config")
                self.assertEqual(graph.prefill.backend, Backend.DISABLED)
        args = self._resolve(enable_prefill_cp=False, cp_strategy=None)
        self.assertFalse(resolution_result(args, "enable_cp_tp_group_sharing"))

    def test_qwen3_next_prefill_cp_is_not_enabled(self):
        with self.assertRaisesRegex(
            ValueError, "not integrated with Qwen3NextForCausalLM"
        ):
            self._resolve("Qwen3NextForCausalLM")

    def test_unsupported_execution_paths_fail_at_startup(self):
        for overrides, message in (
            ({"cp_strategy": "interleave"}, "cp-strategy"),
            ({"enable_mixed_chunk": True}, "enable-mixed-chunk"),
            ({"attn_dp_size": 2}, "attn-dp-size"),
            ({"moe_dp_size": 2}, "moe-dp-size"),
            ({"dcp_size": 2}, "dcp-size"),
            ({"tp_size": 64, "attn_cp_size": 64}, "num_heads"),
            ({"attn_cp_size": 2}, "attn-cp-size == --tp-size"),
            ({"architecture": "UnsupportedKimiLinearForCausalLM"}, "not integrated"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._resolve(**overrides)

    def test_k3_fa4_requires_blackwell_absorbed_mla_geometry(self):
        platform_path = "sglang.srt.arg_groups.cp_tp_group_sharing_hook.get_platform"
        for architecture in (
            "KimiK3LinearForCausalLM",
            "KimiK3ForConditionalGeneration",
        ):
            with (
                self.subTest(architecture=architecture),
                patch(
                    platform_path,
                    return_value=SimpleNamespace(is_sm100_or_sm110=True),
                ),
            ):
                # A split prefill backend takes precedence; NoPE retains the
                # 64 unrotated Q/K dimensions used by the absorbed kernel.
                args = self._resolve(
                    architecture,
                    attention_backend="flashinfer",
                    prefill_attention_backend="fa4",
                    config_overrides={"mla_use_nope": True},
                )
                self.assertTrue(resolution_result(args, "enable_cp_tp_group_sharing"))
                for geometry in (
                    {"kv_lora_rank": 256},
                    {"qk_rope_head_dim": 0},
                    {"qk_rope_head_dim": 128},
                ):
                    with (
                        self.subTest(geometry=geometry),
                        self.assertRaisesRegex(ValueError, "kv_lora_rank=512"),
                    ):
                        self._resolve(
                            architecture,
                            attention_backend="fa4",
                            config_overrides=geometry,
                        )
            with (
                self.subTest(architecture=architecture, blackwell=False),
                patch(
                    platform_path,
                    return_value=SimpleNamespace(is_sm100_or_sm110=False),
                ),
                self.assertRaisesRegex(ValueError, "requires SM100/SM110"),
            ):
                self._resolve(architecture, attention_backend="fa4")

    def test_fa4_cp_dispatches_absorbed_mla_on_blackwell(self):
        from sglang.srt.models.deepseek_common import attention_backend_handler
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
            AttnForwardMethod,
        )

        with (
            patch.object(
                attention_backend_handler,
                "get_platform",
                return_value=SimpleNamespace(is_sm100_or_sm110=True),
            ),
            patch.object(attention_backend_handler, "is_cp_active", return_value=True),
            patch.object(
                attention_backend_handler,
                "is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
            patch.object(
                attention_backend_handler,
                "is_in_breakable_cuda_graph",
                return_value=False,
            ),
            patch.object(
                attention_backend_handler,
                "_dispatch_mla_subtype",
                return_value=AttnForwardMethod.MLA,
            ) as dispatch,
            get_context().override_server_args(enable_deterministic_inference=False),
        ):
            attn, batch = SimpleNamespace(), SimpleNamespace()
            self.assertEqual(
                attention_backend_handler.handle_attention_fa4(attn, batch),
                AttnForwardMethod.MLA,
            )
            dispatch.assert_called_once_with(attn, batch)

    def test_k3_requires_supported_topology_and_prefill_backend(self):
        for architecture in (
            "KimiK3LinearForCausalLM",
            "KimiK3ForConditionalGeneration",
        ):
            for overrides, message in (
                ({"ep_size": 4}, "ep-size"),
                ({"pp_size": 2}, "pp-size"),
                ({"moe_a2a_backend": "deepep"}, "moe-a2a-backend"),
                ({"enable_waterfill": True}, "enable-waterfill"),
                ({"attention_backend": "flashinfer"}, "prefill-attention-backend fa3"),
                (
                    {"prefill_attention_backend": "trtllm_mla"},
                    "prefill-attention-backend fa3",
                ),
                ({"tp_size": 32, "attn_cp_size": 32}, "num_heads=16"),
            ):
                with (
                    self.subTest(architecture=architecture, overrides=overrides),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    self._resolve(architecture, **overrides)

            # The model hook normalizes the unsupported input-scattered mode
            # before the CP hook validates the resolved configuration.
            args = self._resolve(architecture, enable_attn_tp_input_scattered=True)
            self.assertFalse(resolution_result(args, "enable_attn_tp_input_scattered"))

            # A split prefill backend takes precedence over the base backend.
            args = self._resolve(
                architecture,
                attention_backend="flashinfer",
                prefill_attention_backend="fa3",
            )
            self.assertTrue(resolution_result(args, "enable_cp_tp_group_sharing"))


if __name__ == "__main__":
    unittest.main()
