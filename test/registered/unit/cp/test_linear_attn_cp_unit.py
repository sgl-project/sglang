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
                moe.experts = Mock(
                    side_effect=lambda x, topk, rank=rank: x * (rank + 1)
                )
                with (
                    patch.object(
                        ops,
                        "moe_cp_all_gather_into_tensor",
                        side_effect=lambda out, x, gathered=gathered: out.copy_(
                            gathered
                        ),
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
    def test_k3_kda_retains_the_cp_moe_contract_for_both_token_orders(self):
        from sglang.srt.layers.layer_boundary import TokenAxis

        config = SimpleNamespace(
            is_moe=True,
            num_experts=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            routed_expert_hidden_size=4,
            attn_res_block_size=None,
            is_kda_layer=lambda _: True,
        )
        for strategy in ("zigzag", "interleave"):
            parallel = parallel_of(
                attn_dp=1,
                attn_tp=1,
                attn_cp=2,
                enable_prefill_cp=True,
                enable_cp_tp_group_sharing=True,
                cp_strategy=strategy,
            )
            with (
                self.subTest(strategy=strategy),
                get_context().override_server_args(
                    enable_cp_tp_group_sharing=True, cp_strategy=strategy
                ),
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
                attention, feed_forward = build_stages((attn, None), (ffn, None))
                self.assertEqual(attn.in_token_order, strategy == "zigzag")
                self.assertIs(attn.tp_group, SumGroup.TP)
                self.assertTrue(ffn.output_on_cp_shards)
                if strategy == "zigzag":
                    cp_attention = attention.plan.paths[
                        BatchVariant.CONTEXT_PARALLEL
                    ].entry
                    self.assertIs(cp_attention.input_move, ops.cp_gather_in_token_order)
                cp_moe = feed_forward.plan.paths[BatchVariant.CONTEXT_PARALLEL]
                self.assertNotIn(TokenAxis.ATTN_CP, cp_moe.entry.input_rows.sharded)
                self.assertIn(TokenAxis.ATTN_CP, cp_moe.output.layout.sharded)
                self.assertIsNone(cp_moe.output.group)

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
                config.is_kda_layer = lambda index, kda=kda: kda
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


class TestCPModelBoundaries(CustomTestCase):
    def test_models_shard_once_before_layers_and_gather_after_final_norm(self):
        from sglang.srt.models import kimi_linear

        full = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        positions = torch.arange(8)
        local, local_positions = full[::2], positions[::2]
        for module, model_cls in ((kimi_linear, kimi_linear.KimiLinearModel),):
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

    def test_k31_retains_main_interleave_and_ep_support(self):
        from sglang.srt.arg_groups import cp_tp_group_sharing_hook as hook

        cfg = SimpleNamespace(
            enable_prefill_cp=True,
            attn_cp_size=4,
            tp_size=4,
            cp_strategy="interleave",
            speculative_algorithm=None,
            enable_dsa_cache_layer_split=False,
            dcp_size=1,
            enable_mixed_chunk=False,
            moe_dp_size=1,
            ep_size=4,
            moe_a2a_backend="deepep",
            enable_waterfill=False,
            pp_size=2,
            attn_dp_size=1,
            disaggregation_mode="prefill",
        )
        model = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
        )
        with (
            patch.object(hook, "resolving_view", return_value=cfg),
            patch.object(hook, "resolved_view", return_value=cfg),
            patch.object(
                hook,
                "mambaish_config",
                return_value=SimpleNamespace(linear_attn_config={"num_heads": 16}),
            ),
            patch.object(hook, "declare_resolution") as declare,
        ):
            hook.resolve_cp_tp_group_sharing(cfg, model)
        declare.assert_called_once_with(
            cfg, "resolve_cp_tp_group_sharing", enable_cp_tp_group_sharing=True
        )

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
