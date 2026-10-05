"""Replicated attention partitions preserve native projection and loader math."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers import linear
from sglang.srt.layers.layer_boundary.factories import layer_stack
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def build_attention(model, variant, *, width=32, head_dim=8, quant_config=None):
    config = SimpleNamespace(
        hidden_size=width,
        num_attention_heads=8,
        num_key_value_heads=variant[2] if model == "qwen" else 4,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=10000,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        model_type="qwen3_5_text",
        intermediate_size=2 * width,
        hidden_act="silu",
        num_hidden_layers=2,
        attn_output_gate=variant[1] if model == "qwen" else False,
        qk_norm_type="per_head",
        sparse_attention_config=dict(
            sparse_num_index_heads=variant[0], sparse_index_dim=head_dim
        ),
    )
    if model == "qwen":
        from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

        with layer_stack():
            module = Qwen3_5AttentionDecoderLayer(
                config, 0, quant_config=quant_config, is_nextn=variant[0]
            )
        names = ("qkv_proj", "o_proj")
    else:
        from sglang.srt.models.minimax_m3 import MiniMaxM3Attention

        module = MiniMaxM3Attention(
            config,
            quant_config=quant_config,
            is_sparse_attention_layer=True,
            disable_index_value=variant[1],
        )
        names = ("qkv_proj", "o_proj", "index_qkv_proj")
        if not variant[1]:
            names += ("index_o_proj",)
    return module, names


def cases():
    for nextn in (False, True):
        for gate in (False, True):
            for kv_heads in (1, 4):
                yield "qwen", (nextn, gate, kv_heads)
    for heads in (1, 2, 8):
        for disable_value in (False, True):
            yield "minimax", (heads, disable_value)


def values(rows, columns, layer, offset=0):
    return (
        (torch.arange(rows * columns, device=layer.weight.device) + offset) % 29 - 14
    ).reshape(rows, columns).to(layer.weight.dtype) / 128


def load_projection(layer):
    rank, size = layer.tp_rank, layer.tp_size
    with get_parallel().override(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
        if isinstance(layer, linear.QKVParallelLinear):
            shards = []
            for i, name in enumerate(("q", "k", "v")):
                heads = (
                    layer.total_num_heads if name == "q" else layer.total_num_kv_heads
                )
                dim = layer.v_head_size if name == "v" else layer.head_size
                full = values(heads * dim, layer.input_size, layer, i)
                r, s = (
                    (rank, size)
                    if name == "q"
                    else (layer.kv_tp_rank, layer.kv_tp_size)
                )
                pieces = s if name == "q" else min(heads, s)
                index = r if name == "q" else r // max(s // heads, 1)
                layer.weight.weight_loader(layer.weight, full, name)
                shards.append(full.chunk(pieces)[index])
            return torch.cat(shards), None
        full = values(layer.output_size, layer.input_size, layer)
        layer.weight.weight_loader(layer.weight, full)
        return full.chunk(size, dim=1)[rank], None


class TestReplicatedAttentionParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_native_models_keep_independent_kv_and_index_partitions(self):
        for dp in (1, 2):
            for dcp in (1, 2, 4):
                if dcp > 4 // dp:
                    continue
                for rank in range(4):
                    reset_context()
                    publish(
                        ServerArgs(
                            model_path="dummy",
                            device="cpu",
                            tp_size=4,
                            attn_dp_size=dp,
                            dcp_size=dcp,
                        ),
                        role="test",
                        ranks=SpawnRanks(world_rank=rank),
                    )
                    for model, variant in cases():
                        if model == "minimax" and dcp != 1:
                            continue
                        with self.subTest(
                            dp=dp, dcp=dcp, rank=rank, model=model, variant=variant
                        ):
                            module, names = build_attention(model, variant)
                            attn_rank, attn_size = rank % (4 // dp), 4 // dp
                            for name in names:
                                layer = getattr(module, name)
                                if name.startswith("index_"):
                                    size = min(attn_size, variant[0])
                                    expected_rank = attn_rank // (attn_size // size)
                                else:
                                    size, expected_rank = attn_size, attn_rank
                                self.assertEqual(
                                    (layer.tp_rank, layer.tp_size),
                                    (expected_rank, size),
                                )
                                if isinstance(layer, linear.QKVParallelLinear):
                                    factor = (
                                        dcp if model == "qwen" and not variant[0] else 1
                                    )
                                    self.assertEqual(
                                        (layer.kv_tp_rank, layer.kv_tp_size),
                                        (expected_rank // factor, size // factor),
                                    )
                                shard, _ = load_projection(layer)
                                torch.testing.assert_close(layer.weight, shard)
                                x = values(2, shard.shape[1], layer, 7)
                                tp, attn = Mock(), Mock()
                                with (
                                    get_parallel().override(
                                        tp_group=tp, attn_tp_group=attn
                                    ),
                                    patch.object(
                                        linear,
                                        "use_symmetric_memory",
                                        return_value=nullcontext(),
                                    ) as allocation,
                                ):
                                    actual = layer(x)[0]
                                torch.testing.assert_close(actual, F.linear(x, shard))
                                row = isinstance(layer, linear.RowParallelLinear)
                                if row:
                                    self.assertFalse(layer.reduce_results)
                                    self.assertFalse(layer.use_dp_attention_reduce)
                                    self.assertIs(allocation.call_args.args[0], tp)
                                else:
                                    allocation.assert_not_called()
                                self.assertEqual(
                                    tp.all_reduce.call_count
                                    + attn.all_reduce.call_count,
                                    0,
                                )
                                if name.startswith("index_"):
                                    self.assertEqual(
                                        module.idx_replica_size, attn_size // size
                                    )

    def test_group_recipes_freeze_independent_query_and_kv_partitions(self):
        from sglang.srt.layers.linear import ReplicatedParallelGroup

        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=8, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=7),
        )
        for group, factor, rank, size in (
            ("tp", 2, 3, 4),
            ("tp", 4, 1, 2),
            ("attn_tp", 2, 1, 2),
            ("attn_tp", 4, 0, 1),
        ):
            selection = ReplicatedParallelGroup(group, factor)
            self.assertEqual(
                linear.resolve_linear_parallel_group(selection), (rank, size)
            )
            qkv = linear.QKVParallelLinear(
                8,
                2,
                8,
                4,
                bias=False,
                parallel_group="attn_tp",
                kv_parallel_group=selection,
            )
            self.assertEqual((qkv.tp_rank, qkv.tp_size), (3, 4))
            self.assertEqual((qkv.kv_tp_rank, qkv.kv_tp_size), (rank, size))
            expected, _ = load_projection(qkv)
            torch.testing.assert_close(qkv.weight, expected)
            x = values(2, 8, qkv, 3)
            torch.testing.assert_close(qkv(x)[0], F.linear(x, expected))

    def test_invalid_replica_factors_and_conflicting_kv_placement(self):
        from sglang.srt.layers.linear import ReplicatedParallelGroup

        for factor in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ReplicatedParallelGroup("attn_tp", factor)
        with self.assertRaises(ValueError):
            ReplicatedParallelGroup("invalid", 1)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        with self.assertRaises(AssertionError):
            linear.resolve_linear_parallel_group(ReplicatedParallelGroup("tp", 3))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            linear.QKVParallelLinear(8, 2, 8, 4, kv_parallel_group="tp", kv_tp_size=4)


if __name__ == "__main__":
    unittest.main()
