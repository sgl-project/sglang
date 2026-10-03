"""Linear partitions, reloads, and communication follow the selected group."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
                with get_parallel().override(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                    layer.weight.weight_loader(layer.weight, self.weight + 1)
                torch.testing.assert_close(layer.weight, shard + 1)

        with get_parallel().override(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
            with get_parallel().override(tp_rank=1, attn_tp_rank=1):
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
                with get_parallel().override(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
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
                    with get_parallel().override(
                        tp_rank=0, attn_dp_rank=0, attn_tp_rank=0
                    ):
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
        with get_parallel().override(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
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
        with get_parallel().override(tp_group=tp, attn_tp_group=attn):
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
            get_parallel().override(tp_group=tp, attn_tp_group=attn),
            patch(
                "sglang.srt.layers.linear.use_symmetric_memory",
                return_value=nullcontext(),
            ) as allocator,
        ):
            layer = RowParallelLinear(
                8, 8, input_is_parallel=False, parallel_group="attn_tp"
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
            with get_parallel().override(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
                layer.weight.weight_loader(layer.weight, self.weight + 1)
            torch.testing.assert_close(layer.weight, self.weight[:, 4:] + 1)

    def test_replicated_layers_need_no_group_handle(self):
        with (
            get_parallel().override(tp_group=None, attn_tp_group=None),
            patch(
                "sglang.srt.layers.linear.use_symmetric_memory",
                side_effect=AssertionError("unexpected allocator"),
            ),
        ):
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

    def test_legacy_arguments_remain_available_and_mixing_is_rejected(self):
        constructors = (
            lambda **kwargs: ColumnParallelLinear(8, 8, bias=False, **kwargs),
            lambda **kwargs: MergedColumnParallelLinear(
                8, [8, 4], bias=False, **kwargs
            ),
            lambda **kwargs: QKVParallelLinear(8, 2, 4, bias=False, **kwargs),
            lambda **kwargs: RowParallelLinear(8, 8, bias=False, **kwargs),
        )
        for build in constructors:
            self.assertEqual((build().tp_rank, build().tp_size), (3, 4))
            explicit = build(tp_rank=0, tp_size=1)
            self.assertEqual((explicit.tp_rank, explicit.tp_size), (0, 1))
            for kwargs in (dict(tp_rank=0), dict(tp_size=1)):
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    build(parallel_group="replicated", **kwargs)
            with self.assertRaisesRegex(ValueError, "Unknown linear parallel_group"):
                build(parallel_group="unknown")
        for old_reduce in (False, True):
            with self.assertRaisesRegex(ValueError, "use_dp_attention_reduce"):
                RowParallelLinear(
                    8, 8, parallel_group="attn_tp", use_dp_attention_reduce=old_reduce
                )
        legacy = RowParallelLinear(
            8, 8, tp_rank=1, tp_size=2, use_dp_attention_reduce=True
        )
        self.assertTrue(legacy.use_dp_attention_reduce)
        legacy.use_dp_attention_reduce = False
        self.assertFalse(legacy.use_dp_attention_reduce)


if __name__ == "__main__":
    unittest.main()
