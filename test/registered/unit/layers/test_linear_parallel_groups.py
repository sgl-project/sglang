"""Linear partitions, reloads, and communication follow the selected group."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    ReplicatedParallelGroup,
    RowParallelLinear,
)
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.runtime_context import (
    SpawnRanks,
    derive_parallel_widths,
    get_parallel,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestLinearParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        self.x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
        self.weight = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64

    def test_column_partitions_and_reload_use_the_construction_scope(self):
        for group, rank, size in (
            ("tp", 3, 4),
            ("attn_tp", 1, 2),
            ("replicated", 0, 1),
        ):
            with self.subTest(group=group):
                layer = ColumnParallelLinear(8, 8, bias=False, parallel_group=group)
                shard = self.weight.chunk(size, dim=0)[rank]
                layer.weight.weight_loader(layer.weight, self.weight)
                torch.testing.assert_close(layer(self.x)[0], F.linear(self.x, shard))
                with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                    layer.weight.weight_loader(layer.weight, self.weight + 1)
                torch.testing.assert_close(layer.weight, shard + 1)

        with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
            with parallel_scope(tp_rank=1, attn_tp_rank=1):
                layer = ColumnParallelLinear(8, 8, bias=False, parallel_group="attn_tp")
        layer.weight.weight_loader(layer.weight, self.weight)
        torch.testing.assert_close(layer.weight, self.weight[4:])

    def test_packed_loaders_partition_each_logical_weight(self):
        merged = MergedColumnParallelLinear(
            8, [8, 4], bias=False, parallel_group="attn_tp"
        )
        smaller = self.weight[:4] + 2
        merged.weight.weight_loader(merged.weight, self.weight, 0)
        merged.weight.weight_loader(merged.weight, smaller, 1)
        expected = torch.cat((self.weight[4:], smaller[2:]))
        torch.testing.assert_close(merged(self.x)[0], F.linear(self.x, expected))

        for kv_heads in (1, 4):
            with self.subTest(kv_heads=kv_heads):
                qkv = QKVParallelLinear(
                    8, 2, 4, kv_heads, bias=False, parallel_group="attn_tp"
                )
                q, k, v = (
                    self.weight,
                    self.weight[: kv_heads * 2] + 2,
                    self.weight[: kv_heads * 2] + 4,
                )
                for name, weight in (("q", q), ("k", k), ("v", v)):
                    qkv.weight.weight_loader(qkv.weight, weight, name)
                kv_slice = (
                    slice(None) if kv_heads == 1 else slice(k.shape[0] // 2, None)
                )
                expected = torch.cat((q[4:], k[kv_slice], v[kv_slice]))
                torch.testing.assert_close(qkv(self.x)[0], F.linear(self.x, expected))
                with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                    for name, weight in (("q", q), ("k", k), ("v", v)):
                        qkv.weight.weight_loader(qkv.weight, weight + 1, name)
                torch.testing.assert_close(qkv.weight, expected + 1)

    def test_model_qkv_reload_keeps_attention_shards_and_replicated_kv(self):
        from sglang.srt.models.qwen2_moe import Qwen2MoeAttention

        for kv_heads in (1, 4):
            for bias in (False, True):
                with self.subTest(kv_heads=kv_heads, bias=bias):
                    attention = Qwen2MoeAttention(
                        hidden_size=8,
                        num_heads=4,
                        num_kv_heads=kv_heads,
                        max_position_embeddings=16,
                        qkv_bias=bias,
                    )
                    qkv = attention.qkv_proj
                    weights = {
                        "q": self.weight,
                        "k": self.weight[: kv_heads * 2] + 2,
                        "v": self.weight[: kv_heads * 2] + 4,
                    }
                    biases = {
                        name: torch.arange(weight.shape[0], dtype=torch.float32)
                        for name, weight in weights.items()
                    }
                    # Reload under another replica's scope. This model was
                    # constructed on attention rank 1, so Q stays on that rank
                    # and the one-head K/V layout remains replicated.
                    with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                        for name, weight in weights.items():
                            qkv.weight.weight_loader(qkv.weight, weight, name)
                            if bias:
                                qkv.bias.weight_loader(qkv.bias, biases[name], name)
                    shards = {
                        name: value.chunk(2)[1]
                        if name == "q" or kv_heads >= 2
                        else value
                        for name, value in weights.items()
                    }
                    bias_shards = {
                        name: value.chunk(2)[1]
                        if name == "q" or kv_heads >= 2
                        else value
                        for name, value in biases.items()
                    }
                    expected_weight = torch.cat(tuple(shards.values()))
                    expected_bias = (
                        torch.cat(tuple(bias_shards.values())) if bias else None
                    )
                    torch.testing.assert_close(
                        qkv(self.x)[0],
                        F.linear(self.x, expected_weight, expected_bias),
                    )
                    self.assertEqual(attention.q_size, shards["q"].shape[0])
                    self.assertEqual(attention.kv_size, shards["k"].shape[0])

    def test_exaone_attention_output_keeps_tp_allocation_and_partial_results(self):
        from sglang.srt.models.exaone_moe import ExaoneMoEAttention

        tp = SimpleNamespace(world_size=4, all_reduce=Mock())
        attn = SimpleNamespace(world_size=2, all_reduce=Mock())
        for bias in (False, True):
            for symmetric in (False, True):
                with self.subTest(bias=bias, symmetric=symmetric):
                    attention = ExaoneMoEAttention(
                        config=SimpleNamespace(
                            rms_norm_eps=1e-6, layer_types=["full_attention"]
                        ),
                        hidden_size=8,
                        num_heads=4,
                        num_kv_heads=1,
                        max_position_embeddings=16,
                        bias=bias,
                    )
                    row = attention.o_proj
                    with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                        row.weight.weight_loader(row.weight, self.weight)
                        if bias:
                            row.bias.weight_loader(row.bias, torch.arange(8).float())
                    self.assertEqual(rank_size(row), (1, 2))
                    self.assertEqual(attention.q_size, row.input_size_per_partition)
                    self.assertFalse(row.reduce_results)
                    # The boundary reduces this partial result. Its allocation
                    # still uses TP and the DP-padding condition.
                    with (
                        parallel_scope(tp_group=tp, attn_tp_group=attn),
                        patch(
                            "sglang.srt.layers.linear.is_allocation_symmetric",
                            return_value=symmetric,
                        ),
                        patch(
                            "sglang.srt.layers.linear.use_symmetric_memory",
                            return_value=nullcontext(),
                        ) as allocation,
                    ):
                        torch.testing.assert_close(
                            row(self.x[:, 4:])[0],
                            F.linear(self.x[:, 4:], self.weight[:, 4:]),
                        )
                        allocation.assert_called_once_with(tp, disabled=not symmetric)
                    tp.all_reduce.assert_not_called()
                    attn.all_reduce.assert_not_called()

    def test_attention_rows_keep_their_existing_execution_policy(self):
        from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
        from sglang.srt.models.nemotron_h import NemotronHAttention
        from sglang.srt.models.qwen2_moe import Qwen2MoeAttention

        initialize_dp_attention_flags(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2)
        )
        tp = SimpleNamespace(world_size=4, all_reduce=Mock())
        attn = SimpleNamespace(world_size=2, all_reduce=Mock())
        for use_attention_allocation in (False, True):
            with self.subTest(use_attention_allocation=use_attention_allocation):
                if use_attention_allocation:
                    attention = NemotronHAttention(
                        config=SimpleNamespace(
                            hidden_size=8,
                            num_attention_heads=4,
                            num_key_value_heads=1,
                            head_dim=2,
                            sliding_window=None,
                        ),
                        layer_idx=0,
                    )
                else:
                    attention = Qwen2MoeAttention(
                        hidden_size=8,
                        num_heads=4,
                        num_kv_heads=1,
                        max_position_embeddings=16,
                    )
                row = attention.o_proj
                with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                    row.weight.weight_loader(row.weight, self.weight)
                self.assertEqual(rank_size(row), (1, 2))
                self.assertFalse(row.reduce_results)
                self.assertEqual(row.use_dp_attention_reduce, use_attention_allocation)
                with (
                    parallel_scope(tp_group=tp, attn_tp_group=attn),
                    patch(
                        "sglang.srt.layers.linear.is_allocation_symmetric",
                        return_value=False,
                    ),
                    patch(
                        "sglang.srt.layers.linear.use_symmetric_memory",
                        return_value=nullcontext(),
                    ) as allocation,
                ):
                    torch.testing.assert_close(
                        row(self.x[:, 4:])[0],
                        F.linear(self.x[:, 4:], self.weight[:, 4:]),
                    )
                    if use_attention_allocation:
                        allocation.assert_called_once_with(attn)
                    else:
                        allocation.assert_called_once_with(tp, disabled=True)
                tp.all_reduce.assert_not_called()
                attn.all_reduce.assert_not_called()

    def test_model_gate_projections_reload_on_their_attention_shards(self):
        from sglang.srt.models.laguna import LagunaAttention
        from sglang.srt.models.step3p5 import Step3p5Attention

        for model, mode in (
            ("laguna", "disabled"),
            ("laguna", "per-head"),
            ("laguna", "per-element"),
            ("step3p5", False),
            ("step3p5", True),
        ):
            for kv_heads in (1, 4):
                with self.subTest(model=model, mode=mode, kv_heads=kv_heads):
                    options = dict(
                        hidden_size=8,
                        num_heads=4,
                        num_kv_heads=kv_heads,
                        head_dim=2,
                        layer_id=0,
                        rms_norm_eps=1e-6,
                        rope_theta=10000,
                        rope_scaling=None,
                        partial_rotary_factor=1.0,
                        max_position_embeddings=16,
                    )
                    if model == "laguna":
                        attention = LagunaAttention(
                            **options,
                            attention_bias=True,
                            sliding_window_size=-1,
                            layer_type="full_attention",
                            gating=mode,
                        )
                    else:
                        attention = Step3p5Attention(
                            **options, use_head_wise_attn_gate=mode
                        )
                    qkv = attention.qkv_proj
                    weights = {
                        "q": self.weight,
                        "k": self.weight[: kv_heads * 2] + 2,
                        "v": self.weight[: kv_heads * 2] + 4,
                    }
                    with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                        for name, weight in weights.items():
                            qkv.weight.weight_loader(qkv.weight, weight, name)
                            if qkv.bias is not None:
                                qkv.bias.weight_loader(
                                    qkv.bias, torch.zeros(weight.shape[0]), name
                                )
                    shards = [
                        weight.chunk(2)[1] if name == "q" or kv_heads >= 2 else weight
                        for name, weight in weights.items()
                    ]
                    torch.testing.assert_close(
                        qkv(self.x)[0], F.linear(self.x, torch.cat(shards))
                    )
                    gate = getattr(attention, "g_proj", None)
                    if mode in (False, "disabled"):
                        self.assertIsNone(gate)
                        continue
                    gate_weight = (
                        self.weight if mode == "per-element" else self.weight[:4]
                    )
                    with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                        gate.weight.weight_loader(gate.weight, gate_weight)
                    torch.testing.assert_close(
                        gate(self.x)[0], F.linear(self.x, gate_weight.chunk(2)[1])
                    )

    def test_v2_loaders_keep_their_partition_during_reload(self):
        column = ColumnParallelLinear(8, 8, bias=False, parallel_group="attn_tp")
        row = RowParallelLinear(8, 8, bias=False, parallel_group="attn_tp")
        merged = MergedColumnParallelLinear(
            8, [8, 4], bias=False, parallel_group="attn_tp"
        )
        qkv = QKVParallelLinear(8, 2, 4, 1, bias=False, parallel_group="attn_tp")
        for layer in (column, row, merged, qkv):
            layer.weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight_loader_v2,
            )
        # Reload while a different rank is current. The stored rank still owns
        # the latter half of Q and each MLP matrix, while K/V stay replicated.
        with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
            column.weight.weight_loader(column.weight, self.weight)
            row.weight.weight_loader(row.weight, self.weight)
            merged.weight.weight_loader(merged.weight, self.weight, 0)
            merged.weight.weight_loader(merged.weight, self.weight[:4] + 2, 1)
            for name, weight in (
                ("q", self.weight),
                ("k", self.weight[:2] + 2),
                ("v", self.weight[:2] + 4),
            ):
                qkv.weight.weight_loader(qkv.weight, weight, name)
        torch.testing.assert_close(column.weight, self.weight[4:])
        torch.testing.assert_close(row.weight, self.weight[:, 4:])
        torch.testing.assert_close(
            merged.weight, torch.cat((self.weight[4:], self.weight[2:4] + 2))
        )
        torch.testing.assert_close(
            qkv.weight,
            torch.cat((self.weight[4:], self.weight[:2] + 2, self.weight[:2] + 4)),
        )

    def test_gather_uses_the_selected_group(self):
        tp = SimpleNamespace(world_size=4, all_gather=Mock())
        attn = SimpleNamespace(world_size=2, all_gather=Mock())
        with parallel_scope(tp_group=tp, attn_tp_group=attn):
            for group, rank, size, selected, other in (
                ("tp", 3, 4, tp, attn),
                ("attn_tp", 1, 2, attn, tp),
            ):
                with self.subTest(group=group):
                    selected.all_gather.reset_mock()
                    other.all_gather.reset_mock()
                    expected = F.linear(self.x, self.weight)
                    selected.all_gather.return_value = expected
                    layer = ColumnParallelLinear(
                        8, 8, bias=False, gather_output=True, parallel_group=group
                    )
                    layer.weight.weight_loader(layer.weight, self.weight)
                    torch.testing.assert_close(layer(self.x)[0], expected)
                    payload, dim = selected.all_gather.call_args.args
                    torch.testing.assert_close(
                        payload, expected.chunk(size, dim=-1)[rank]
                    )
                    self.assertEqual(dim, -1)
                    other.all_gather.assert_not_called()

    def test_row_uses_attention_group_and_preserves_reduction_overrides(self):
        bias = torch.arange(8, dtype=torch.float32)
        # Rank 2 contributes the bias once; rank 3 supplies the other input shard.
        peer = F.linear(self.x[:, :4], self.weight[:, :4], bias)
        partial = F.linear(self.x[:, 4:], self.weight[:, 4:])
        attn = SimpleNamespace(
            world_size=2, all_reduce=Mock(side_effect=lambda x: x + peer)
        )
        tp = SimpleNamespace(
            world_size=4, all_reduce=Mock(side_effect=AssertionError("wrong group"))
        )
        with (
            parallel_scope(tp_group=tp, attn_tp_group=attn),
            patch(
                "sglang.srt.layers.linear.use_symmetric_memory",
                return_value=nullcontext(),
            ) as allocator,
        ):
            layer = RowParallelLinear(
                8,
                8,
                input_is_parallel=False,
                parallel_group="attn_tp",
                use_dp_attention_reduce=True,
            )
            layer.weight.weight_loader(layer.weight, self.weight)
            layer.bias.weight_loader(layer.bias, bias)
            torch.testing.assert_close(
                layer(self.x)[0], F.linear(self.x, self.weight, bias)
            )
            torch.testing.assert_close(attn.all_reduce.call_args.args[0], partial)
            allocator.assert_called_with(attn)
            tp.all_reduce.assert_not_called()
            attn.all_reduce.reset_mock()
            torch.testing.assert_close(layer(self.x, skip_all_reduce=True)[0], partial)
            attn.all_reduce.assert_not_called()
            layer.reduce_results = False
            torch.testing.assert_close(layer(self.x)[0], partial)
            attn.all_reduce.assert_not_called()
            with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                layer.weight.weight_loader(layer.weight, self.weight + 1)
            torch.testing.assert_close(layer.weight, self.weight[:, 4:] + 1)

    def test_row_partition_keeps_independent_reduction_and_allocation_policies(self):
        tp = SimpleNamespace(world_size=4, all_reduce=Mock(side_effect=lambda x: x * 2))
        attn = SimpleNamespace(
            world_size=2, all_reduce=Mock(side_effect=lambda x: x * 3)
        )
        for group, rank, size in (
            ("tp", 3, 4),
            ("attn_tp", 1, 2),
            ("replicated", 0, 1),
        ):
            default = RowParallelLinear(8, 8, parallel_group=group)
            self.assertFalse(default.use_dp_attention_reduce)
            for attention_reduce in (False, True):
                for reduce_results in (False, True):
                    for symmetric in (False, True):
                        with self.subTest(
                            group=group,
                            attention_reduce=attention_reduce,
                            reduce_results=reduce_results,
                            symmetric=symmetric,
                        ):
                            layer = RowParallelLinear(
                                8,
                                8,
                                bias=False,
                                input_is_parallel=False,
                                parallel_group=group,
                                use_dp_attention_reduce=attention_reduce,
                                reduce_results=reduce_results,
                            )
                            with parallel_scope(
                                tp_rank=0, attn_dp_rank=0, attn_tp_rank=0
                            ):
                                layer.weight.weight_loader(layer.weight, self.weight)
                            shard = self.weight.chunk(size, dim=1)[rank]
                            partial = F.linear(self.x.chunk(size, dim=1)[rank], shard)
                            with (
                                parallel_scope(tp_group=tp, attn_tp_group=attn),
                                patch(
                                    "sglang.srt.layers.linear.is_allocation_symmetric",
                                    return_value=symmetric,
                                ),
                                patch(
                                    "sglang.srt.layers.linear.use_symmetric_memory",
                                    return_value=nullcontext(),
                                ) as allocator,
                            ):
                                # Existing runtime writers may switch execution
                                # policy without changing the stored partition.
                                for policy in (attention_reduce, not attention_reduce):
                                    layer.use_dp_attention_reduce = policy
                                    tp.all_reduce.reset_mock()
                                    attn.all_reduce.reset_mock()
                                    allocator.reset_mock()
                                    selected, other = (
                                        (attn, tp) if policy else (tp, attn)
                                    )
                                    reduces = reduce_results and size > 1
                                    expected = (
                                        partial * (3 if policy else 2)
                                        if reduces
                                        else partial
                                    )
                                    torch.testing.assert_close(
                                        layer(self.x)[0], expected
                                    )
                                    if reduces:
                                        selected.all_reduce.assert_called_once()
                                        torch.testing.assert_close(
                                            selected.all_reduce.call_args.args[0],
                                            partial,
                                        )
                                    else:
                                        selected.all_reduce.assert_not_called()
                                    other.all_reduce.assert_not_called()
                                    if policy:
                                        allocator.assert_called_once_with(attn)
                                    else:
                                        allocator.assert_called_once_with(
                                            tp, disabled=not symmetric
                                        )
                                    self.assertEqual(rank_size(layer), (rank, size))

    def test_replicated_model_mlps_keep_weights_and_tp_output_allocation(self):
        from sglang.srt.models.exaone_moe import ExaoneMoEMLP
        from sglang.srt.models.laguna import LagunaMLP
        from sglang.srt.models.nemotron_h import NemotronHMLP
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
        from sglang.srt.models.step3p5 import Step3p5MLP

        tp = SimpleNamespace(
            world_size=4, all_reduce=Mock(side_effect=lambda tensor: tensor * 2)
        )
        for cls in (
            ExaoneMoEMLP,
            LagunaMLP,
            NemotronHMLP,
            Qwen2MoeMLP,
            Step3p5MLP,
        ):
            groups = [(None, 3, 4), ("tp", 3, 4), ("replicated", 0, 1)]
            for group, rank, size in groups:
                with self.subTest(model=cls.__name__, group=group):
                    tp.all_reduce.reset_mock()
                    reduces = group is None
                    options = dict(
                        intermediate_size=8,
                        reduce_results=reduces,
                        parallel_group=group,
                    )
                    if cls is NemotronHMLP:
                        options["config"] = SimpleNamespace(hidden_size=8)
                    else:
                        options["hidden_size"] = 8
                        if cls is not Step3p5MLP:
                            options["hidden_act"] = "silu"
                    if group is None:
                        options.pop("parallel_group")
                    mlp = cls(**options)
                    # Use native activation for CPU weight and layout checks.
                    mlp.act_fn._forward_method = mlp.act_fn.forward_native
                    up = getattr(mlp, "gate_up_proj", getattr(mlp, "up_proj", None))
                    down = mlp.down_proj
                    with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                        if cls is NemotronHMLP:
                            up.weight.weight_loader(up.weight, self.weight)
                        else:
                            up.weight.weight_loader(up.weight, self.weight, 0)
                            up.weight.weight_loader(up.weight, self.weight + 1, 1)
                        down.weight.weight_loader(down.weight, self.weight + 2)
                    local_weight = self.weight.chunk(size)[rank]
                    projected = F.linear(self.x, local_weight)
                    if cls is NemotronHMLP:
                        activated = projected.relu().square()
                    else:
                        activated = F.silu(projected) * F.linear(
                            self.x, (self.weight + 1).chunk(size)[rank]
                        )
                    expected = F.linear(
                        activated, (self.weight + 2).chunk(size, dim=1)[rank]
                    )
                    if reduces:
                        expected = expected * 2
                    with (
                        parallel_scope(tp_group=tp),
                        patch(
                            "sglang.srt.layers.linear.is_allocation_symmetric",
                            return_value=True,
                        ),
                        patch(
                            "sglang.srt.layers.linear.use_symmetric_memory",
                            return_value=nullcontext(),
                        ) as allocator,
                    ):
                        torch.testing.assert_close(mlp(self.x), expected)
                        allocator.assert_called_once_with(tp, disabled=False)
                    if reduces:
                        tp.all_reduce.assert_called_once()
                    else:
                        tp.all_reduce.assert_not_called()
                    self.assertEqual(rank_size(up), (rank, size))
                    self.assertEqual(rank_size(down), (rank, size))

    def test_step_shared_expert_constructs_a_replicated_mlp(self):
        from sglang.srt.models import step3p5

        config = SimpleNamespace(
            hidden_size=8,
            layer_types=["full_attention"],
            yarn_only_types=[],
            rope_theta=[10000],
            max_position_embeddings=16,
            head_dim=2,
            moe_layers_enum="0",
            num_attention_heads=4,
            num_attention_groups=1,
            num_hidden_layers=1,
            swiglu_limits_shared=None,
            partial_rotary_factors=[1.0],
            rms_norm_eps=1e-6,
            use_head_wise_attn_gate=False,
            share_expert_dim=8,
        )
        backend = Mock()
        backend.is_deepep.return_value = True
        with (
            patch.object(step3p5, "get_moe_a2a_backend", return_value=backend),
            patch.object(step3p5, "Step3p5MoEMLP", return_value=torch.nn.Identity()),
            patch.object(step3p5, "append_stages", return_value=(Mock(), Mock())),
        ):
            layer = step3p5.Step3p5DecoderLayer(config)
        self.assertEqual(rank_size(layer.share_expert.gate_up_proj)[1], 1)
        self.assertEqual(rank_size(layer.share_expert.down_proj)[1], 1)
        self.assertFalse(layer.share_expert.down_proj.reduce_results)

    def test_replicated_layers_need_no_group_handle(self):
        with parallel_scope(tp_group=None, attn_tp_group=None):
            for cls, kwargs in (
                (ColumnParallelLinear, dict(gather_output=True)),
                (RowParallelLinear, dict(input_is_parallel=False)),
            ):
                with self.subTest(layer=cls.__name__):
                    layer = cls(8, 8, bias=False, parallel_group="replicated", **kwargs)
                    layer.weight.weight_loader(layer.weight, self.weight)
                    torch.testing.assert_close(
                        layer(self.x)[0], F.linear(self.x, self.weight)
                    )

    def test_constructor_placement_accepts_groups_and_rejects_integer_arguments(self):
        constructors = (
            lambda **kwargs: ColumnParallelLinear(8, 8, bias=False, **kwargs),
            lambda **kwargs: MergedColumnParallelLinear(
                8, [8, 4], bias=False, **kwargs
            ),
            lambda **kwargs: QKVParallelLinear(8, 2, 4, bias=False, **kwargs),
            lambda **kwargs: RowParallelLinear(8, 8, bias=False, **kwargs),
            lambda **kwargs: MergedColumnParallelRepeatedLinear(
                8, [8, 4], [2], **kwargs
            ),
            lambda **kwargs: ColumnParallelBatchedLinear(
                2, 8, 8, torch.float32, **kwargs
            ),
        )
        for build in constructors:
            self.assertEqual(rank_size(build()), (3, 4))
            replicated = build(parallel_group="replicated")
            self.assertEqual(rank_size(replicated), (0, 1))
            for kwargs in (dict(tp_rank=0), dict(tp_size=1)):
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                    build(**kwargs)
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                    build(parallel_group="replicated", **kwargs)
            with self.assertRaisesRegex(ValueError, "Unknown linear parallel_group"):
                build(parallel_group="unknown")
        for old_reduce in (False, True):
            layer = RowParallelLinear(
                8, 8, parallel_group="attn_tp", use_dp_attention_reduce=old_reduce
            )
            self.assertEqual(rank_size(layer), (1, 2))
            self.assertEqual(layer.use_dp_attention_reduce, old_reduce)
        layer = RowParallelLinear(
            8, 8, parallel_group="attn_tp", use_dp_attention_reduce=True
        )
        self.assertTrue(layer.use_dp_attention_reduce)
        layer.use_dp_attention_reduce = False
        self.assertFalse(layer.use_dp_attention_reduce)

    def test_layers_own_groups_without_rank_or_size_snapshots(self):
        group = get_parallel().attn_tp_group
        layer = QKVParallelLinear(8, 2, 4, 1, parallel_group="attn_tp")
        self.assertIs(layer.tp_group, group)
        self.assertIs(layer.kv_tp_group, group)
        for key in ("tp_rank", "tp_size", "kv_tp_rank", "kv_tp_size"):
            self.assertFalse(hasattr(layer, key))
        with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
            self.assertIs(layer.tp_group, group)
            self.assertIsNot(layer.tp_group, get_parallel().attn_tp_group)
            self.assertEqual(rank_size(layer), (1, 2))
        with parallel_scope(attn_tp_group=None):
            offline = ColumnParallelLinear(8, 8, bias=False, parallel_group="attn_tp")
        with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
            offline.weight.weight_loader(offline.weight, self.weight)
        torch.testing.assert_close(offline.weight, self.weight[4:])
        self.assertEqual(rank_size(offline), (1, 2))

    def test_lora_slicing_reads_retained_query_and_kv_groups(self):
        from sglang.srt.lora.layers import (
            ColumnParallelLinearWithLoRA,
            MergedColumnParallelLinearWithLoRA,
            QKVParallelLinearWithLoRA,
            RowParallelLinearWithLoRA,
        )

        column = ColumnParallelLinear(8, 8, bias=False)
        merged = MergedColumnParallelLinear(8, [8, 4], bias=False)
        qkv = QKVParallelLinear(
            8,
            2,
            4,
            2,
            bias=False,
            parallel_group="tp",
            kv_parallel_group=ReplicatedParallelGroup("tp", 2),
        )
        row = RowParallelLinear(8, 8, bias=False, reduce_results=False)
        for cls, layer, rows in (
            (ColumnParallelLinearWithLoRA, column, [6, 7]),
            (MergedColumnParallelLinearWithLoRA, merged, [6, 7, 11]),
            (QKVParallelLinearWithLoRA, qkv, [6, 7, 10, 11, 14, 15]),
        ):
            with self.subTest(wrapper=cls.__name__):
                wrapped = cls.__new__(cls)
                torch.nn.Module.__init__(wrapped)
                wrapped.base_layer = layer
                if isinstance(layer, QKVParallelLinear):
                    full_rows = (
                        layer.total_num_heads + 2 * layer.total_num_kv_heads
                    ) * layer.head_size
                else:
                    full_rows = layer.output_size
                full = torch.arange(full_rows * 2, dtype=torch.float32).reshape(-1, 2)
                with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
                    actual = wrapped.slice_lora_b_weights(full)
                torch.testing.assert_close(actual, full[rows], rtol=0, atol=0)
        wrapped = RowParallelLinearWithLoRA.__new__(RowParallelLinearWithLoRA)
        torch.nn.Module.__init__(wrapped)
        wrapped.base_layer = row
        full = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
            actual = wrapped.slice_lora_a_weights(full)
        torch.testing.assert_close(actual, full[:, 6:], rtol=0, atol=0)

    def test_quant_initialization_keeps_the_entry_scope_partition(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        observed = []

        class ScopeChangingConfig:
            def get_quant_method(config, layer, prefix):
                observed.append(
                    (rank_size(layer)[0], rank_size(layer)[1], layer.parallel_group)
                )
                scope = parallel_scope(
                    tp_size=1,
                    tp_rank=0,
                    attn_tp_rank=0,
                    attn_dp_rank=0,
                    moe_tp_rank=0,
                    **derive_parallel_widths(
                        tp_size=1,
                        attn_cp_size=1,
                        attn_dp_size=1,
                        moe_ep_size=1,
                        moe_dp_size=1,
                        dcp_size=1,
                        dcp_enabled=False,
                    ),
                )
                scope.__enter__()
                method = UnquantizedLinearMethod()
                native_create = method.create_weights

                def create_weights(**kwargs):
                    try:
                        native_create(**kwargs)
                    finally:
                        scope.__exit__(None, None, None)

                method.create_weights = create_weights
                return method

        for group, rank, size in (
            (None, 3, 4),
            ("attn_tp", 1, 2),
            ("replicated", 0, 1),
        ):
            config = ScopeChangingConfig()
            options = dict(bias=False, quant_config=config, parallel_group=group)
            layers = (
                ColumnParallelLinear(8, 8, **options),
                MergedColumnParallelLinear(8, [8, 4], **options),
                QKVParallelLinear(
                    8,
                    2,
                    4,
                    1,
                    kv_parallel_group=ReplicatedParallelGroup("attn_tp", 2),
                    **options,
                ),
                RowParallelLinear(8, 8, reduce_results=False, **options),
                MergedColumnParallelRepeatedLinear(
                    8, [8, 4], [2], quant_config=config, parallel_group=group
                ),
            )
            self.assertEqual(observed[-5:], [(rank, size, group)] * 5)
            self.assertEqual(get_parallel().tp_size, 4)
            for layer in layers:
                with self.subTest(group=group, layer=type(layer).__name__):
                    self.assertEqual(rank_size(layer), (rank, size))
                    if isinstance(layer, RowParallelLinear):
                        expected = self.weight.chunk(size, dim=1)[rank]
                        layer.weight.weight_loader(layer.weight, self.weight)
                    elif isinstance(layer, QKVParallelLinear):
                        pieces = (
                            ("q", self.weight),
                            ("k", self.weight[:2] + 2),
                            ("v", self.weight[:2] + 4),
                        )
                        for shard, weight in pieces:
                            layer.weight.weight_loader(layer.weight, weight, shard)
                        expected = torch.cat(
                            (self.weight.chunk(size)[rank], pieces[1][1], pieces[2][1])
                        )
                        self.assertEqual(rank_size(layer, kv=True), (0, 1))
                    elif isinstance(
                        layer,
                        (
                            MergedColumnParallelLinear,
                            MergedColumnParallelRepeatedLinear,
                        ),
                    ):
                        pieces = [self.weight, self.weight[:4] + 2]
                        if isinstance(layer, MergedColumnParallelRepeatedLinear):
                            pieces.append(self.weight[:2] + 4)
                        for shard, weight in enumerate(pieces):
                            layer.weight.weight_loader(layer.weight, weight, shard)
                        expected = torch.cat(
                            [
                                weight.chunk(size)[rank] if i < 2 else weight
                                for i, weight in enumerate(pieces)
                            ]
                        )
                    else:
                        expected = self.weight.chunk(size)[rank]
                        layer.weight.weight_loader(layer.weight, self.weight)
                    torch.testing.assert_close(layer.weight, expected)


if __name__ == "__main__":
    unittest.main()
