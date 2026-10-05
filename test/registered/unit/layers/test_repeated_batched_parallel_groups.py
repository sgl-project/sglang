"""Repeated and batched projections retain native checkpoint partitions."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers import linear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

CASES = (
    ("glm", "fused"),
    ("glm", "bfg"),
    ("glm", "split"),
    ("kimi", "fused"),
    ("kimi", "split"),
    ("kimi", "full"),
    ("kimi", "full_fp8"),
)


def build_projections(
    kind, route, shard_attn=False, *, width=8, heads=4, head_dim=4, reduce=False
):
    from sglang.srt.models.glm5_next import Glm5NextLinearAttention
    from sglang.srt.models.kimi_linear import KimiDeltaAttention

    prefix = "model.layers.0.self_attn"
    config = SimpleNamespace(
        dtype=torch.get_default_dtype(),
        linear_attn_config=dict(
            head_dim=head_dim, num_heads=heads, short_conv_kernel_size=3
        ),
    )
    quant = None
    if route in ("bfg", "split", "full_fp8"):
        ignored = (
            [
                prefix + "." + n
                for n in ("b_proj", "f_a_proj", "g_a_proj", "f_b_proj", "g_b_proj")
            ]
            if route == "bfg"
            else []
        )
        quant = Fp8Config(
            ignored_layers=ignored,
            packed_modules_mapping=Glm5NextLinearAttention._PACKED_MODULES_MAPPING
            if kind == "glm"
            else None,
        )
    cls = Glm5NextLinearAttention if kind == "glm" else KimiDeltaAttention
    kwargs = (
        {}
        if kind == "glm"
        else dict(no_kda_lora=route.startswith("full"), shard_on_attn_tp=shard_attn)
    )
    module = cls(
        0,
        width,
        config,
        quant_config=quant,
        prefix=prefix,
        reduce_results=reduce,
        **kwargs,
    )
    classes = (
        linear.ColumnParallelLinear,
        linear.RowParallelLinear,
        linear.MergedColumnParallelRepeatedLinear,
        linear.ColumnParallelBatchedLinear,
        linear.ReplicatedLinear,
    )
    names = tuple(
        name for name, layer in module.named_children() if isinstance(layer, classes)
    )
    return module, names


def values(rows, columns, layer, offset=0):
    return (
        (torch.arange(rows * columns, device=layer.weight.device) + offset) % 23 - 11
    ).reshape(rows, columns).to(layer.weight.dtype) / 128


def load_projection(layer):
    rank, size = getattr(layer, "tp_rank", 0), getattr(layer, "tp_size", 1)
    with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
        if isinstance(layer, linear.MergedColumnParallelRepeatedLinear):
            shards = []
            for i, local_size in enumerate(layer.output_partition_sizes):
                column = i < layer.num_column_parallel
                full = values(
                    local_size * size if column else local_size,
                    layer.input_size,
                    layer,
                    i,
                )
                layer.weight.weight_loader(layer.weight, full, i)
                shards.append(full.chunk(size)[rank] if column else full)
            return torch.cat(shards)
        if isinstance(layer, linear.ColumnParallelBatchedLinear):
            shards = []
            for i in range(layer.weight.shape[0]):
                full = values(
                    layer.weight.shape[1] * size, layer.weight.shape[2], layer, i
                )
                layer.weight.weight_loader(layer.weight, full, i)
                shards.append(full.chunk(size)[rank])
            return torch.stack(shards)
        if isinstance(layer, linear.QKVParallelLinear):
            shards = []
            for i, name in enumerate(("q", "k", "v")):
                heads = (
                    layer.total_num_heads if name == "q" else layer.total_num_kv_heads
                )
                dim = layer.v_head_size if name == "v" else layer.head_size
                full = values(heads * dim, layer.input_size, layer, i)
                layer.weight.weight_loader(layer.weight, full, name)
                shards.append(full.chunk(size)[rank])
            return torch.cat(shards)
        full = values(layer.output_size, layer.input_size, layer)
        if layer.weight.ndim == 3:
            full = full.unsqueeze(1)
        layer.weight.weight_loader(layer.weight, full)
        if isinstance(layer, linear.RowParallelLinear):
            return full.chunk(size, dim=1)[rank]
        return torch.cat(
            [
                part.chunk(size)[rank]
                for part in full.split(
                    getattr(layer, "output_sizes", [layer.output_size])
                )
            ]
        )


class TestRepeatedBatchedParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.float32)

    def test_helper_group_selection_frozen_reload_and_native_math(self):
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for group, rank, size in (
            (None, 3, 4),
            ("tp", 3, 4),
            ("attn_tp", 1, 2),
            ("replicated", 0, 1),
        ):
            for cls in (
                linear.MergedColumnParallelRepeatedLinear,
                linear.ColumnParallelBatchedLinear,
            ):
                kwargs = {} if group is None else dict(parallel_group=group)
                layer = (
                    cls(8, [16, 8], [4], **kwargs)
                    if cls is linear.MergedColumnParallelRepeatedLinear
                    else cls(2, 8, 16, dtype=torch.float32, **kwargs)
                )
                expected = load_projection(layer)
                torch.testing.assert_close(layer.weight, expected)
                self.assertEqual((layer.tp_rank, layer.tp_size), (rank, size))
                if cls is linear.MergedColumnParallelRepeatedLinear:
                    x = values(2, 8, layer, 3)
                    actual = layer(x)
                    reference = F.linear(x, expected)
                else:
                    x = values(4, 8, layer, 3).reshape(2, 2, 8)
                    actual = layer(x)
                    reference = torch.stack(
                        [F.linear(x[i], expected[i]) for i in range(2)]
                    )
                torch.testing.assert_close(actual, reference)

    def test_native_attention_projection_routes_loaders_and_row_policy(self):
        for dp in (1, 2):
            for rank in (0, 3):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                tp, attn = Mock(), Mock()
                tp.all_reduce.side_effect = lambda x: x * 4
                for kind, route in CASES:
                    for shard_attn in (False, True) if kind == "kimi" else (True,):
                        for reduce in (False, True):
                            with self.subTest(
                                dp=dp,
                                rank=rank,
                                kind=kind,
                                route=route,
                                shard_attn=shard_attn,
                                reduce=reduce,
                            ):
                                module, names = build_projections(
                                    kind, route, shard_attn, reduce=reduce
                                )
                                for name in names:
                                    layer = getattr(module, name)
                                    expected = load_projection(layer)
                                    torch.testing.assert_close(layer.weight, expected)
                                    conv = name == "qkv_conv1d"
                                    batch = isinstance(
                                        layer, linear.ColumnParallelBatchedLinear
                                    )
                                    row = isinstance(layer, linear.RowParallelLinear)
                                    size = (
                                        4 // dp
                                        if kind == "glm"
                                        or shard_attn
                                        or isinstance(layer, linear.QKVParallelLinear)
                                        else 4
                                    )
                                    if isinstance(layer, linear.ReplicatedLinear):
                                        size = 1
                                    local_rank = rank % size
                                    self.assertEqual(
                                        (
                                            getattr(layer, "tp_rank", 0),
                                            getattr(layer, "tp_size", 1),
                                        ),
                                        (local_rank, size),
                                    )
                                    if conv:
                                        x = values(
                                            2 * expected.shape[0], 6, layer, 3
                                        ).reshape(2, expected.shape[0], 6)
                                        reference = (
                                            x.unfold(-1, 3, 1)
                                            * expected.squeeze(1)[None, :, None, :]
                                        ).sum(-1)
                                        actual = F.conv1d(
                                            x, layer.weight, groups=expected.shape[0]
                                        )
                                    elif batch:
                                        x = values(
                                            2 * expected.shape[0],
                                            expected.shape[2],
                                            layer,
                                            3,
                                        ).reshape(expected.shape[0], 2, -1)
                                        reference = torch.stack(
                                            [
                                                F.linear(x[i], expected[i])
                                                for i in range(expected.shape[0])
                                            ]
                                        )
                                        actual = layer(x)
                                    else:
                                        x = values(2, expected.shape[1], layer, 3)
                                        reference = F.linear(x, expected)
                                        if row and reduce and size > 1:
                                            reference *= 4
                                        tp.all_reduce.reset_mock()
                                        attn.all_reduce.reset_mock()
                                        dense = (
                                            patch.object(
                                                layer.quant_method,
                                                "apply",
                                                side_effect=lambda l, x, bias=None: (
                                                    F.linear(x, l.weight, bias)
                                                ),
                                            )
                                            if layer.quant_method.__class__.__name__
                                            == "Fp8LinearMethod"
                                            else nullcontext()
                                        )
                                        with (
                                            parallel_scope(
                                                tp_group=tp, attn_tp_group=attn
                                            ),
                                            patch(
                                                "sglang.srt.layers.linear.use_symmetric_memory",
                                                return_value=nullcontext(),
                                            ) as allocation,
                                            patch(
                                                "sglang.srt.layers.linear.is_allocation_symmetric",
                                                return_value=False,
                                            ),
                                            dense,
                                        ):
                                            out = layer(x)
                                            actual = (
                                                out[0]
                                                if isinstance(out, tuple)
                                                else out
                                            )
                                            if row:
                                                allocation.assert_called_once_with(
                                                    tp, disabled=True
                                                )
                                            else:
                                                allocation.assert_not_called()
                                        self.assertEqual(
                                            tp.all_reduce.call_count,
                                            int(row and reduce and size > 1),
                                        )
                                        attn.all_reduce.assert_not_called()
                                    torch.testing.assert_close(actual, reference)


if __name__ == "__main__":
    unittest.main()
