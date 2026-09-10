"""Regressions for complete sequences and matching head shards under prefill CP."""

import json
import unittest
import weakref
from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch
from torch import nn

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
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
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)
from sglang.srt.models.kimi_k3 import (
    KimiK3DecoderLayer,
    KimiK3DeltaAttention,
    KimiK3MLAAttention,
)
from sglang.srt.models.kimi_linear import KimiDecoderLayer, KimiDeltaAttention, KimiMoE
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
_KIMI = "sglang.srt.models.kimi_linear."
_K3 = "sglang.srt.models.kimi_k3."
_COMM = "sglang.srt.layers.communicator."
_ZIGZAG = "sglang.srt.layers.cp.zigzag."
_MLA = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla."


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

    def test_k3_mla_slices_before_projection_and_restores_ragged_output(self):
        positions = torch.arange(len(self.full))
        expected = self.full * 3 + positions[:, None]
        for rank, batch in enumerate(self.batches):
            with self.subTest(rank=rank):
                group = Mock()

                def gather(output, local):
                    rows = output.shape[0] // 2
                    shards = [
                        self.strategy.shard_hidden_states(expected, other)[:rows]
                        for other in self.batches
                    ]
                    torch.cat(shards, out=output)

                group.all_gather_into_tensor.side_effect = gather
                layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
                nn.Module.__init__(layer)
                layer.is_kda_layer = False
                layer.self_attn = Mock(
                    side_effect=lambda hidden_states, positions, **kwargs: hidden_states
                    * 3
                    + positions[:, None]
                )
                context = Mock()
                with (
                    get_parallel().override(attn_cp_rank=rank, attn_cp_group=group),
                    patch(_COMM + "AttentionInputs") as inputs,
                    patch(_COMM + "get_attn_tp_context", return_value=context),
                ):
                    output = layer._run_self_attn_inner(
                        self.full, positions, batch, None
                    )
                torch.testing.assert_close(output, expected)
                local_hidden, local_batch, prepare = inputs.call_args.args
                torch.testing.assert_close(local_hidden, self.shards[rank])
                self.assertIs(local_batch, batch)
                self.assertIs(prepare, layer.self_attn.prepare_qkv_latent)
                torch.testing.assert_close(
                    layer.self_attn.call_args.kwargs["positions"],
                    self.strategy.shard_position_ids(positions, batch),
                )
                group.all_gather_into_tensor.assert_called_once()
                group.all_reduce.assert_not_called()
                context.clear_attn_inputs.assert_called_once()

    def test_k3_kda_and_inactive_mla_preserve_full_rows(self):
        for kda, mode, length in (
            (True, ForwardMode.EXTEND, len(self.full)),
            (False, ForwardMode.EXTEND, 3),
            (False, ForwardMode.DECODE, 3),
        ):
            with self.subTest(kda=kda, mode=mode):
                hidden, positions = self.full[:length], torch.arange(length)
                batch = (
                    self.batches[0]
                    if kda
                    else SimpleNamespace(
                        input_ids=positions,
                        forward_mode=mode,
                        extend_seq_lens_cpu=[length],
                        attn_cp_metadata=None,
                    )
                )
                layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
                nn.Module.__init__(layer)
                layer.is_kda_layer = kda
                layer.self_attn = Mock(
                    side_effect=lambda hidden_states, **kw: hidden_states * 2
                )
                layer.self_attn.prepare_qkv_latent = None
                group = Mock()
                with get_parallel().override(attn_cp_group=group):
                    output = layer._run_self_attn_inner(hidden, positions, batch, None)
                self.assertIs(layer.self_attn.call_args.kwargs["hidden_states"], hidden)
                torch.testing.assert_close(output, hidden * 2)
                group.all_gather_into_tensor.assert_not_called()

    def test_k3_failed_mla_clears_lazy_projection_context(self):
        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.is_kda_layer = False
        layer.self_attn = Mock(side_effect=RuntimeError("attention failed"))
        context = Mock()
        with (
            patch(_COMM + "AttentionInputs"),
            patch(_COMM + "get_attn_tp_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "attention failed"),
        ):
            layer._run_self_attn_inner(
                self.full, torch.arange(len(self.full)), self.batches[0], None
            )
        context.clear_attn_inputs.assert_called_once()

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

        for enabled, cp_size in ((False, 1), (False, 2), (True, 2), (True, 8)):
            for full_rank_gate in (False, True):
                with (
                    self.subTest(
                        enabled=enabled, cp_size=cp_size, full_rank_gate=full_rank_gate
                    ),
                    ExitStack() as stack,
                ):
                    stack.enter_context(
                        get_context().override_server_args(
                            enable_linear_attn_cp=enabled
                        )
                    )
                    stack.enter_context(
                        get_parallel().override(
                            tp_size=8,
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
                    stack.enter_context(patch(_K3 + "k3_gemm_ar.maybe_wrap_o_proj"))
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
                                (call.kwargs["tp_rank"], call.kwargs["tp_size"]),
                                (rank, size),
                            )
                    output_args = mocks["RowParallelLinear"].call_args.kwargs
                    self.assertTrue(output_args["reduce_results"])
                    self.assertEqual(
                        output_args["use_dp_attention_reduce"], not enabled
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


class TestKimiK3CPKVOverlap(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.strategy = ZigzagCPStrategy(cp_size=2)
        self.batch = self._batch()
        self.layer = SimpleNamespace(layer_id=1)
        self.key = torch.arange(9, dtype=torch.float32).reshape(3, 1, 3)
        self.rope = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
        self.main, self.comm = Mock(), Mock()
        self.events = []

        def make_event():
            event = Mock()
            self.events.append(event)
            return event

        self.group = Mock()
        self.group.pynccl_comm.available = True
        self.group.pynccl_comm.change_state.side_effect = lambda **kw: nullcontext()
        self.enterContext(
            get_parallel().override(attn_cp_group=self.group, attn_cp_rank=0)
        )
        # Execute real packing/cache-write control flow on CPU; replace only
        # CUDA scheduling and the remote-rank data returned by the collective.
        self.enterContext(
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            )
        )
        self.enterContext(patch.object(torch.Tensor, "record_stream"))
        self.enterContext(
            patch(
                _ZIGZAG + "torch.cuda.is_current_stream_capturing", return_value=False
            )
        )
        self.enterContext(
            patch(_ZIGZAG + "torch.cuda.current_stream", return_value=self.main)
        )
        self.stream_factory = self.enterContext(
            patch(_ZIGZAG + "torch.cuda.Stream", return_value=self.comm)
        )
        self.enterContext(
            patch(
                _ZIGZAG + "torch.cuda.stream", side_effect=lambda stream: nullcontext()
            )
        )
        self.enterContext(patch(_ZIGZAG + "torch.cuda.Event", side_effect=make_event))
        self.enterContext(
            patch(
                _ZIGZAG + "use_symmetric_memory",
                side_effect=lambda *a, **kw: nullcontext(),
            )
        )
        self.gather = self.group.all_gather_into_tensor
        self.gather.side_effect = lambda output, local: output.copy_(
            torch.cat([local, local])
        )
        self.pool = Mock()
        self.enterContext(
            patch(_ZIGZAG + "get_token_to_kv_pool", return_value=self.pool)
        )

    @staticmethod
    def _batch():
        return SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(
                pending_mla_kv_materializations={},
                per_rank_actual_token=[3, 3],
                per_rank_logical_token=None,
                # One six-token sequence split into four zigzag blocks.
                reverse_split_len=[2, 1, 2, 1],
                cp_reverse_index=[0, 2, 3, 1],
            ),
            out_cache_loc=torch.arange(6),
        )

    @staticmethod
    def _expected_full(local):
        # Both ranks return this test tensor; interleave the real zigzag blocks.
        return torch.cat([local[:2], local[:2], local[2:], local[2:]])

    def test_backend_joins_once_and_pending_kv_is_layer_and_batch_local(self):
        self.assertTrue(
            self.strategy.start_mla_kv_materialization(
                self.batch, self.layer, self.key, self.rope
            )
        )
        self.comm.wait_stream.assert_called_once_with(self.main)
        self.main.wait_event.assert_not_called()
        self.assertEqual(self.events, [])
        self.pool.set_mla_kv_buffer.assert_not_called()
        pending = self.batch.attn_cp_metadata.pending_mla_kv_materializations[1]
        self.assertIs(pending.inputs[0], self.key)
        self.assertIs(pending.inputs[2], self.batch.out_cache_loc)
        self.assertIs(pending.stream, self.comm)
        self.assertIs(pending.gather.output_buffer, self.gather.call_args.args[0])
        self.assertIs(pending.gather.send_buffer, self.gather.call_args.args[1])

        other_layer, next_batch = SimpleNamespace(layer_id=3), self._batch()
        self.assertFalse(
            self.strategy.finish_mla_kv_materialization(next_batch, self.layer)
        )
        self.assertFalse(
            self.strategy.finish_mla_kv_materialization(self.batch, other_layer)
        )
        # A synchronous layer must not consume another layer's pending write.
        self.strategy.materialize_full_mla_kv(
            self.batch, other_layer, self.key + 10, self.rope
        )
        self.main.wait_event.assert_not_called()
        self.assertEqual(self.gather.call_count, 2)
        self.strategy.materialize_full_mla_kv(
            self.batch, self.layer, self.key, self.rope
        )
        self.main.wait_event.assert_called_once_with(self.events[0])
        self.events[0].record.assert_called_once_with(self.comm)
        # Only the producer dependency is issued; finish must not wait for Q.
        self.comm.wait_stream.assert_called_once_with(self.main)
        self.assertEqual(self.gather.call_count, 2)
        self.assertEqual(self.pool.set_mla_kv_buffer.call_count, 2)
        first_write = self.pool.set_mla_kv_buffer.call_args.args
        self.assertIs(first_write[0], self.layer)
        self.assertIs(first_write[1], self.batch.out_cache_loc)
        torch.testing.assert_close(first_write[2], self._expected_full(self.key))
        torch.testing.assert_close(first_write[3], self._expected_full(self.rope))
        self.assertFalse(self.batch.attn_cp_metadata.pending_mla_kv_materializations)
        self.assertFalse(
            self.strategy.finish_mla_kv_materialization(self.batch, self.layer)
        )

        self.assertTrue(
            self.strategy.start_mla_kv_materialization(
                next_batch, self.layer, self.key + 20, self.rope
            )
        )
        self.strategy.materialize_full_mla_kv(
            next_batch, self.layer, self.key, self.rope
        )
        self.assertEqual(self.gather.call_count, 3)
        self.assertEqual(self.pool.set_mla_kv_buffer.call_count, 3)
        self.main.wait_event.assert_called_with(self.events[1])
        torch.testing.assert_close(
            self.pool.set_mla_kv_buffer.call_args.args[2],
            self._expected_full(self.key + 20),
        )

    def test_saved_producer_event_excludes_queued_query_work(self):
        ready = Mock()
        self.strategy.start_mla_kv_materialization(
            self.batch, self.layer, self.key, self.rope, producer_event=ready
        )
        self.comm.wait_event.assert_called_once_with(ready)
        self.comm.wait_stream.assert_not_called()
        pending = self.batch.attn_cp_metadata.pending_mla_kv_materializations[1]
        self.assertIs(pending.producer_event, ready)
        self.strategy.finish_mla_kv_materialization(self.batch, self.layer)
        self.comm.wait_stream.assert_not_called()
        self.main.wait_event.assert_called_once_with(self.events[0])
        torch.testing.assert_close(
            self.pool.set_mla_kv_buffer.call_args.args[2],
            self._expected_full(self.key),
        )

    def test_prepared_transfer_launches_once_after_buffer_preparation(self):
        ready = Mock()
        self.assertTrue(
            self.strategy.start_mla_kv_materialization(
                self.batch,
                self.layer,
                self.key,
                self.rope,
                producer_event=ready,
                prepare_only=True,
            )
        )
        self.gather.assert_not_called()
        self.pool.set_mla_kv_buffer.assert_not_called()
        pending = self.batch.attn_cp_metadata.pending_mla_kv_materializations[1]
        self.assertFalse(pending.launched)
        self.assertIs(pending.group, self.group)
        send, output = pending.gather.send_buffer, pending.gather.output_buffer
        self.assertTrue(
            self.strategy.launch_mla_kv_materialization(self.batch, self.layer)
        )
        self.assertTrue(pending.launched)
        self.gather.assert_called_once_with(output, send)
        # Retrying a launch must not issue a second collective.
        self.assertTrue(
            self.strategy.launch_mla_kv_materialization(self.batch, self.layer)
        )
        self.gather.assert_called_once()
        self.assertTrue(
            self.strategy.finish_mla_kv_materialization(self.batch, self.layer)
        )
        self.comm.wait_event.assert_called_once_with(ready)
        self.comm.wait_stream.assert_not_called()
        torch.testing.assert_close(
            self.pool.set_mla_kv_buffer.call_args.args[2], self._expected_full(self.key)
        )

    def test_unlaunched_preparation_is_drained_without_cache_write(self):
        self.strategy.start_mla_kv_materialization(
            self.batch,
            self.layer,
            self.key,
            self.rope,
            prepare_only=True,
        )
        self.assertFalse(
            self.strategy.finish_mla_kv_materialization(self.batch, self.layer)
        )
        self.main.wait_stream.assert_called_once_with(self.comm)
        self.gather.assert_not_called()
        self.pool.set_mla_kv_buffer.assert_not_called()
        self.assertFalse(self.batch.attn_cp_metadata.pending_mla_kv_materializations)

    def test_unlaunched_preparation_falls_back_to_synchronous_materialization(self):
        self.strategy.start_mla_kv_materialization(
            self.batch,
            self.layer,
            self.key,
            self.rope,
            prepare_only=True,
        )
        self.strategy.materialize_full_mla_kv(
            self.batch, self.layer, self.key, self.rope
        )
        self.main.wait_stream.assert_called_once_with(self.comm)
        self.gather.assert_called_once()
        self.pool.set_mla_kv_buffer.assert_called_once()
        torch.testing.assert_close(
            self.pool.set_mla_kv_buffer.call_args.args[2], self._expected_full(self.key)
        )

    def test_query_exception_cleans_unlaunched_preparation(self):
        owner = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
        nn.Module.__init__(owner)
        owner.use_output_gate, owner._cp_kv_overlap = False, True
        owner.attn_mqa = self.layer
        self.strategy.start_mla_kv_materialization(
            self.batch,
            self.layer,
            self.key,
            self.rope,
            prepare_only=True,
        )
        pending = self.batch.attn_cp_metadata.pending_mla_kv_materializations[1]
        staging = [
            weakref.ref(pending.gather.output_buffer),
            weakref.ref(pending.gather.send_buffer),
        ]
        del pending
        self.main.wait_stream.side_effect = lambda stream: self.assertTrue(
            all(ref() is not None for ref in staging)
        )
        with (
            patch(_K3 + "is_cp_active", return_value=True),
            patch(_K3 + "get_cp_strategy", return_value=self.strategy),
            patch(
                _K3 + "DeepseekV2AttentionMLA.forward",
                side_effect=RuntimeError("query failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "query failed"),
        ):
            owner(None, None, self.batch, None)
        self.main.wait_stream.assert_called_once_with(self.comm)
        self.gather.assert_not_called()
        self.pool.set_mla_kv_buffer.assert_not_called()
        self.assertFalse(self.batch.attn_cp_metadata.pending_mla_kv_materializations)

    def test_k3_records_readiness_only_for_eligible_cuda_overlap(self):
        owner = SimpleNamespace(
            _cp_kv_overlap=True, current_attention_backend="fa4", rotary_emb=None
        )
        with (
            patch(_K3 + "is_cp_active", return_value=True),
            patch(_K3 + "get_is_capture_mode", return_value=False),
            patch(_K3 + "is_in_breakable_cuda_graph", return_value=False),
            patch(_K3 + "is_in_tc_piecewise_cuda_graph", return_value=False),
        ):
            ready = KimiK3MLAAttention.record_mla_cp_kv_producer_event(
                owner, self.batch, self.key
            )
            self.assertIs(ready, self.events[0])
            ready.record.assert_called_once_with(self.main)
            owner._cp_kv_overlap = False
            self.assertIsNone(
                KimiK3MLAAttention.record_mla_cp_kv_producer_event(
                    owner, self.batch, self.key
                )
            )
        self.assertEqual(len(self.events), 1)

    def test_k3_forwards_saved_producer_event(self):
        owner = SimpleNamespace(
            _cp_kv_overlap=True,
            current_attention_backend="fa4",
            rotary_emb=None,
            attn_mqa=self.layer,
        )
        ready = Mock()
        with (
            patch(_K3 + "is_cp_active", return_value=True),
            patch(_K3 + "get_is_capture_mode", return_value=False),
            patch(_K3 + "get_cp_strategy", return_value=self.strategy),
            patch.object(self.strategy, "start_mla_kv_materialization") as launch,
        ):
            KimiK3MLAAttention.prepare_mla_cp_kv(
                owner, self.batch, self.key, self.rope, producer_event=ready
            )
        launch.assert_called_once_with(
            self.batch,
            self.layer,
            self.key,
            self.rope,
            producer_event=ready,
            prepare_only=False,
        )

    def test_unsupported_launch_falls_back_to_one_synchronous_write(self):
        for cuda, capture, pynccl in (
            (False, False, Mock(available=True)),
            (True, True, Mock(available=True)),
            (True, False, None),
            (True, False, Mock(available=False)),
        ):
            with (
                self.subTest(cuda=cuda, capture=capture, pynccl=pynccl),
                patch.object(
                    torch.Tensor,
                    "is_cuda",
                    new_callable=PropertyMock,
                    return_value=cuda,
                ),
                patch(
                    _ZIGZAG + "torch.cuda.is_current_stream_capturing",
                    return_value=capture,
                ),
            ):
                self.group.pynccl_comm = pynccl
                batch = self._batch()
                before = self.gather.call_count
                self.assertFalse(
                    self.strategy.start_mla_kv_materialization(
                        batch, self.layer, self.key, self.rope
                    )
                )
                self.strategy.materialize_full_mla_kv(
                    batch, self.layer, self.key, self.rope
                )
                self.assertEqual(self.gather.call_count, before + 1)
                self.assertFalse(batch.attn_cp_metadata.pending_mla_kv_materializations)
        self.stream_factory.assert_not_called()
        self.main.wait_event.assert_not_called()

    def test_failed_async_write_joins_stream_and_leaves_no_pending_state(self):
        self.strategy.start_mla_kv_materialization(
            self.batch, self.layer, self.key, self.rope
        )
        self.pool.set_mla_kv_buffer.side_effect = RuntimeError("cache write failed")
        with self.assertRaisesRegex(RuntimeError, "cache write failed"):
            self.strategy.finish_mla_kv_materialization(self.batch, self.layer)
        self.gather.assert_called_once()
        self.main.wait_stream.assert_called_once_with(self.comm)
        self.assertEqual(self.events, [])
        self.assertFalse(self.batch.attn_cp_metadata.pending_mla_kv_materializations)

    def test_failed_gather_retains_staging_until_stream_is_joined(self):
        staging = []

        def fail_gather(output, send):
            staging.extend((weakref.ref(output), weakref.ref(send)))
            del output, send
            raise RuntimeError("gather launch failed")

        def check_staging(stream):
            self.assertIs(stream, self.comm)
            self.assertTrue(all(reference() is not None for reference in staging))

        self.main.wait_stream.side_effect = check_staging
        with (
            patch.object(self.group, "all_gather_into_tensor", new=fail_gather),
            self.assertRaisesRegex(RuntimeError, "gather launch failed"),
        ):
            self.strategy.start_mla_kv_materialization(
                self.batch, self.layer, self.key, self.rope
            )
        self.main.wait_stream.assert_called_once_with(self.comm)
        self.pool.set_mla_kv_buffer.assert_not_called()
        self.assertEqual(self.events, [])
        self.assertFalse(self.batch.attn_cp_metadata.pending_mla_kv_materializations)

    def test_finish_uses_snapshot_of_locations_and_layout(self):
        self.strategy.start_mla_kv_materialization(
            self.batch, self.layer, self.key, self.rope
        )
        original_loc = self.batch.out_cache_loc
        self.batch.out_cache_loc = torch.arange(6) + 100
        self.batch.attn_cp_metadata.cp_reverse_index[:] = [3, 2, 1, 0]
        self.batch.attn_cp_metadata.reverse_split_len[:] = [1, 2, 1, 2]
        self.batch.attn_cp_metadata.per_rank_actual_token[:] = [1, 5]
        self.strategy.finish_mla_kv_materialization(self.batch, self.layer)
        write = self.pool.set_mla_kv_buffer.call_args.args
        self.assertIs(write[1], original_loc)
        torch.testing.assert_close(write[2], self._expected_full(self.key))

    def test_finish_failure_keeps_gather_buffers_alive_until_join(self):
        self.strategy.start_mla_kv_materialization(
            self.batch, self.layer, self.key, self.rope
        )
        pending = self.batch.attn_cp_metadata.pending_mla_kv_materializations[1]
        staging = [
            weakref.ref(pending.gather.output_buffer),
            weakref.ref(pending.gather.send_buffer),
        ]
        del pending
        self.gather.reset_mock()

        def check_staging(stream):
            self.assertTrue(all(reference() is not None for reference in staging))

        def fail_compact(prepared):
            del prepared
            raise RuntimeError("compaction failed")

        self.main.wait_stream.side_effect = check_staging
        with (
            patch.object(self.strategy, "_compact_all_gather_rows", new=fail_compact),
            self.assertRaisesRegex(RuntimeError, "compaction failed"),
        ):
            self.strategy.finish_mla_kv_materialization(self.batch, self.layer)
        self.main.wait_stream.assert_called_once_with(self.comm)
        self.assertFalse(self.batch.attn_cp_metadata.pending_mla_kv_materializations)

    def test_k3_only_launches_for_eager_no_rope_flash_attention_cp(self):
        owner = SimpleNamespace(attn_mqa=self.layer)
        for enabled, backend, rope, capture, active in (
            (True, "fa3", None, False, True),
            (True, "fa4", None, False, True),
            (False, "fa3", None, False, True),
            (True, "flashinfer", None, False, True),
            (True, "fa3", object(), False, True),
            (True, "fa3", None, True, True),
            (True, "fa3", None, False, False),
        ):
            owner._cp_kv_overlap = enabled
            owner.current_attention_backend = backend
            owner.rotary_emb = rope
            with (
                self.subTest(
                    enabled=enabled,
                    backend=backend,
                    rope=rope,
                    capture=capture,
                    active=active,
                ),
                patch(_K3 + "is_cp_active", return_value=active),
                patch(_K3 + "get_is_capture_mode", return_value=capture),
                patch(_K3 + "get_cp_strategy", return_value=self.strategy),
                patch.object(self.strategy, "start_mla_kv_materialization") as launch,
            ):
                KimiK3MLAAttention.prepare_mla_cp_kv(
                    owner, self.batch, self.key, self.rope
                )
                self.assertEqual(
                    launch.call_count,
                    int(
                        enabled
                        and backend in ("fa3", "fa4")
                        and rope is None
                        and not capture
                        and active
                    ),
                )

    def test_k3_failure_after_launch_drains_pending_cache_write(self):
        owner = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
        nn.Module.__init__(owner)
        owner.use_output_gate, owner._cp_kv_overlap = False, True
        owner.attn_mqa = self.layer
        self.strategy.start_mla_kv_materialization(
            self.batch, self.layer, self.key, self.rope
        )
        with (
            patch(_K3 + "is_cp_active", return_value=True),
            patch(_K3 + "get_cp_strategy", return_value=self.strategy),
            patch(
                _K3 + "DeepseekV2AttentionMLA.forward",
                side_effect=RuntimeError("query failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "query failed"),
        ):
            owner(None, None, self.batch, None)
        self.main.wait_event.assert_called_once_with(self.events[0])
        self.assertFalse(self.batch.attn_cp_metadata.pending_mla_kv_materializations)
        self.gather.assert_called_once()

    def test_shared_mla_records_kv_before_query_and_launches_after_projection(self):
        latent = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        context, stages = Mock(), []
        context.fetch_qkv_latent.return_value = latent
        owner = SimpleNamespace(
            q_lora_rank=2,
            kv_lora_rank=3,
            qk_rope_head_dim=1,
            alt_stream=None,
            use_dsa=False,
            _can_fuse_bmm_into_attention=lambda batch: False,
            _cp_kv_overlap=True,
            kv_a_layernorm=lambda x: x + 20,
        )

        ready = Mock()

        def record_ready(batch, key):
            stages.append("ready")
            self.assertIs(batch, self.batch)
            torch.testing.assert_close(key, latent[:, 2:5] + 20)
            return ready

        owner.record_mla_cp_kv_producer_event = record_ready

        def publish(batch, key, rope, *, producer_event, prepare_only):
            stages.append("prepare")
            self.assertIs(batch, self.batch)
            self.assertIs(producer_event, ready)
            self.assertTrue(prepare_only)
            torch.testing.assert_close(key, (latent[:, 2:5] + 20).unsqueeze(1))
            torch.testing.assert_close(rope, latent[:, 5:].unsqueeze(1))
            return True

        def launch(batch):
            stages.append("launch")
            self.assertIs(batch, self.batch)
            raise RuntimeError("stop before attention kernels")

        owner.launch_mla_cp_kv = launch

        def query_normalization(query):
            stages.append("q_norm")
            return query + 10

        owner.q_a_layernorm = query_normalization

        def query_projection(query):
            stages.append("q_b")
            torch.testing.assert_close(query, latent[:, :2] + 10)
            return query

        owner.prepare_mla_cp_kv, owner.q_b_proj_forward = publish, query_projection
        with (
            get_context().override_server_args(dcp_replicate_q_proj=False),
            patch(_MLA + "get_attn_tp_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "stop before attention kernels"),
        ):
            DeepseekMLAForwardMixin.forward_absorb_prepare(
                owner, None, None, self.batch, None
            )
        self.assertEqual(stages, ["ready", "prepare", "q_norm", "q_b", "launch"])

    def test_shared_mla_query_failure_does_not_start_transfer(self):
        latent = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        context = Mock()
        context.fetch_qkv_latent.return_value = latent
        for failure in ("q_norm", "q_b"):
            stages = []

            def q_norm(query):
                stages.append("q_norm")
                if failure == "q_norm":
                    raise RuntimeError("query failed")
                return query

            def q_b(query):
                stages.append("q_b")
                raise RuntimeError("query failed")

            owner = SimpleNamespace(
                q_lora_rank=2,
                kv_lora_rank=3,
                qk_rope_head_dim=1,
                alt_stream=None,
                use_dsa=False,
                _can_fuse_bmm_into_attention=lambda batch: False,
                _cp_kv_overlap=True,
                kv_a_layernorm=lambda x: x + 20,
                record_mla_cp_kv_producer_event=Mock(
                    side_effect=lambda *a: stages.append("ready") or Mock()
                ),
                prepare_mla_cp_kv=Mock(
                    side_effect=lambda *a, **kw: stages.append("prepare") or True
                ),
                launch_mla_cp_kv=Mock(),
                q_a_layernorm=q_norm,
                q_b_proj_forward=q_b,
            )
            with (
                self.subTest(failure=failure),
                get_context().override_server_args(dcp_replicate_q_proj=False),
                patch(_MLA + "get_attn_tp_context", return_value=context),
                self.assertRaisesRegex(RuntimeError, "query failed"),
            ):
                DeepseekMLAForwardMixin.forward_absorb_prepare(
                    owner, None, None, self.batch, None
                )
            owner.prepare_mla_cp_kv.assert_called_once()
            self.assertTrue(owner.prepare_mla_cp_kv.call_args.kwargs["prepare_only"])
            owner.launch_mla_cp_kv.assert_not_called()
            self.assertEqual(
                stages,
                ["ready", "prepare", "q_norm"] + (["q_b"] if failure == "q_b" else []),
            )


class TestLinearAttnCPResolution(CustomTestCase):
    def _resolve(
        self, architecture="Qwen3NextForCausalLM", *, config_overrides=None, **overrides
    ):
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
                tp_size=4,
                attn_cp_size=2,
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                skip_tokenizer_init=True,
            )
            if kimi_k3:
                args["attention_backend"] = "fa3"
            args.update(overrides)
            server_args = ServerArgs(**args)
            server_args.resolve_once()
            return server_args

    def test_supported_models_enable_cp_and_disable_prefill_graphs(self):
        for architecture in (
            "Qwen3NextForCausalLM",
            "KimiLinearForCausalLM",
            "KimiK3LinearForCausalLM",
            "KimiK3ForConditionalGeneration",
        ):
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
            ({"architecture": "UnsupportedKimiLinearForCausalLM"}, "not integrated"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._resolve(**overrides)

    def test_k3_fa4_requires_blackwell_absorbed_mla_geometry(self):
        platform_path = "sglang.srt.arg_groups.linear_attn_cp_hook.get_platform"
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
                self.assertTrue(resolution_result(args, "enable_linear_attn_cp"))
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
                (
                    {"enable_attn_tp_input_scattered": True},
                    "enable-attn-tp-input-scattered",
                ),
                ({"attention_backend": "flashinfer"}, "prefill-attention-backend fa3"),
                (
                    {"prefill_attention_backend": "trtllm_mla"},
                    "prefill-attention-backend fa3",
                ),
                ({"tp_size": 32}, "num_heads=16"),
            ):
                with (
                    self.subTest(architecture=architecture, overrides=overrides),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    self._resolve(architecture, **overrides)

            # A split prefill backend takes precedence over the base backend.
            args = self._resolve(
                architecture,
                attention_backend="flashinfer",
                prefill_attention_backend="fa3",
            )
            self.assertTrue(resolution_result(args, "enable_linear_attn_cp"))


if __name__ == "__main__":
    unittest.main()
