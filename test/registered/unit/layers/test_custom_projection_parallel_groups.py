"""Custom projections retain frozen checkpoint shards and native math."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers import linear, mova
from sglang.srt.models.inkling_common import dense_mlp
from sglang.srt.models.inkling_common.attn import InklingAttention
from sglang.srt.models.inkling_common.moe import _build_inkling_shared_experts
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def values(shape, offset=0, *, device="cpu", dtype=torch.float32):
    count = 1
    for size in shape:
        count *= size
    return (
        (((torch.arange(count, device=device) + offset) % 29 - 14) / 128)
        .reshape(shape)
        .to(dtype)
    )


def build_attention(kv_heads, *, width=32, head_dim=8, quant_config=None, bias=False):
    return InklingAttention(
        width,
        8,
        kv_heads,
        head_dim,
        head_dim,
        8,
        4,
        1e-6,
        False,
        0,
        q_bias=bias,
        o_bias=bias,
        quant_config=quant_config,
    )


def load_projection(layer):
    weights, biases = [], []
    r, s = rank_size(layer)[0], rank_size(layer)[1]
    row = isinstance(layer, linear.RowParallelLinear)
    with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
        if row:
            full = values(
                (layer.output_size, layer.input_size),
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            layer.weight.weight_loader(layer.weight, full)
            weights.append(full.chunk(s, dim=1)[r])
            if layer.bias is not None:
                full = values(
                    (layer.output_size,),
                    4,
                    device=layer.bias.device,
                    dtype=layer.bias.dtype,
                )
                layer.bias.weight_loader(layer.bias, full)
                biases.append(full)
        else:
            for i, size in enumerate(layer.output_sizes):
                full = values(
                    (size, layer.input_size),
                    i,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                layer.weight.weight_loader(layer.weight, full, i)
                weights.append(full.chunk(s)[r])
                if layer.bias is not None:
                    full = values(
                        (size,), i + 4, device=layer.bias.device, dtype=layer.bias.dtype
                    )
                    layer.bias.weight_loader(layer.bias, full, i)
                    biases.append(full.chunk(s)[r])
    return weights[0] if row else torch.cat(weights), (
        biases[0] if row else torch.cat(biases)
    ) if biases else None


def build_batch(group=None, *, width=16, linearized=False, execution_group=None):
    kwargs = {} if group is None else dict(parallel_group=group)
    return dense_mlp.InklingBatchDenseMLP(
        2,
        width,
        2 * width,
        0,
        "shared",
        tp_group=execution_group,
        linearized_bf16=linearized,
        **kwargs,
    )


def load_batch(module):
    device, dtype = module.w13_weight.device, module.w13_weight.dtype
    full_f = module.intermediate_size_per_partition * module.moe_tp_size
    full = {
        "w1": values((2, full_f, module.hidden_size), 0, device=device, dtype=dtype),
        "w3": values((2, full_f, module.hidden_size), 1, device=device, dtype=dtype),
        "w2": values((2, module.hidden_size, full_f), 2, device=device, dtype=dtype),
    }
    full13 = torch.stack([full["w1"], full["w3"]], dim=2).flatten(1, 2)
    with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
        module.weight_loader_fused(module.w13_weight, full13, "w13.weight", "w13")
        module.weight_loader_fused(module.w2_weight, full["w2"], "w2.weight", "w2")
    shards = {
        name: weight.chunk(module.moe_tp_size, dim=2 if name == "w2" else 1)[
            module.moe_tp_rank
        ]
        for name, weight in full.items()
    }
    expected13 = torch.stack([shards["w1"], shards["w3"]], dim=2).flatten(1, 2)
    torch.testing.assert_close(module.w13_weight, expected13)
    torch.testing.assert_close(module.w2_weight, shards["w2"])
    return shards


def batch_reference(x, gamma, shards):
    outputs = []
    for i in range(gamma.shape[-1]):
        gate = F.linear(x, shards["w1"][i])
        up = F.linear(x, shards["w3"][i])
        activation = (
            F.silu(gate.float()) * up.float() * gamma[:, i : i + 1].float()
        ).to(x.dtype)
        outputs.append(F.linear(activation, shards["w2"][i]))
    return torch.stack(outputs).float().sum(0).to(x.dtype)


def build_values(group="attn_tp", *, width=16):
    return mova.RoutedValueExperts(2, width, width, parallel_group=group)


def load_values(module, rank, size):
    full = values(
        (2, module.output_size, module.input_size),
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
        module.weight_loader(module.weight, full)
    expected = full.chunk(size, dim=1)[rank]
    torch.testing.assert_close(module.weight, expected)
    return expected


class TestCustomProjectionParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_native_custom_loaders_and_math_after_construction_scope(self):
        for dp in (1, 2):
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for kv_heads in (1, 4):
                    module = build_attention(kv_heads, bias=True)
                    self.assertEqual(rank_size(module.qkvr)[1], 4 // dp)
                    self.assertEqual(module.qkvr.inkling_num_kv_heads, kv_heads)
                    for name in ("qkvr", "wo_ud"):
                        layer = getattr(module, name)
                        expected, bias = load_projection(layer)
                        torch.testing.assert_close(layer.weight, expected)
                        x = values((2, expected.shape[1]))
                        used_bias = (
                            bias if name == "qkvr" or rank_size(layer)[0] == 0 else None
                        )
                        with (
                            parallel_scope(tp_group=Mock(), attn_tp_group=Mock()),
                            patch.object(
                                linear,
                                "use_symmetric_memory",
                                return_value=nullcontext(),
                            ),
                        ):
                            actual = layer(x)[0]
                        torch.testing.assert_close(
                            actual, F.linear(x, expected, used_bias)
                        )
                for group, r, s in (
                    (None, 0, 1),
                    ("replicated", 0, 1),
                    ("tp", rank, 4),
                    ("attn_tp", rank % (4 // dp), 4 // dp),
                ):
                    for linearized in (False, True):
                        module = build_batch(group, linearized=linearized)
                        self.assertEqual(
                            (module.moe_tp_rank, module.moe_tp_size), (r, s)
                        )
                        shards = load_batch(module)
                        x, gamma = values((2, 16)), values((2, 2), 8) + 1
                        eager_sum = getattr(
                            dense_mlp._sum_dim0,
                            "_torchdynamo_orig_callable",
                            dense_mlp._sum_dim0,
                        )

                        def swiglu(y, g):
                            return (
                                F.silu(y[..., ::2].float())
                                * y[..., 1::2].float()
                                * g.unsqueeze(-1).float()
                            ).to(y.dtype)

                        with (
                            patch.object(module, "_swiglu", swiglu),
                            patch.object(dense_mlp, "_sum_dim0", eager_sum),
                        ):
                            actual = module(x, gamma)
                        torch.testing.assert_close(
                            actual, batch_reference(x, gamma, shards)
                        )
                        self.assertEqual(module.moe_runner_config.moe_tp_rank, r)
                        self.assertEqual(module.helper.moe_tp_rank, r)
                        self.assertEqual(module.helper.moe_tp_size, s)
                    value_module = build_values(group or "replicated")
                    expected = load_values(value_module, r, s)
                    selected = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
                    gamma = values((2, 2), 9) + 1
                    x = values((2, 16))
                    actual = value_module(x, gamma, selected)
                    reference = torch.stack(
                        [
                            sum(
                                F.silu(F.linear(x[t], expected[e])) * gamma[t, i]
                                for i, e in enumerate(selected[t].tolist())
                            )
                            for t in range(2)
                        ]
                    )
                    torch.testing.assert_close(actual, reference)

    def test_shared_factory_keeps_full_tp_and_execution_handle(self):
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        execution = Mock()
        with (
            parallel_scope(tp_group=execution),
            patch(
                "sglang.srt.models.inkling_common.moe.use_inkling_shared_fused_moe",
                return_value=False,
            ),
        ):
            module = _build_inkling_shared_experts(
                n_shared_experts=2,
                shared_expert_sink=True,
                shared_experts_size=1,
                inference_moe_w13_interleaved=True,
                hidden_size=16,
                intermediate_size=32,
                layer_id=0,
                prefix="model.layers.0",
            )
        self.assertEqual((module.moe_tp_rank, module.moe_tp_size), (3, 4))
        self.assertIs(module.tp_group, execution)
        shards = load_batch(module)
        x, gamma = values((2, 16)), values((2, 2), 8) + 1
        eager_sum = getattr(
            dense_mlp._sum_dim0, "_torchdynamo_orig_callable", dense_mlp._sum_dim0
        )

        def swiglu(y, g):
            return (
                F.silu(y[..., ::2].float())
                * y[..., 1::2].float()
                * g.unsqueeze(-1).float()
            ).to(y.dtype)

        with (
            patch.object(module, "_swiglu", swiglu),
            patch.object(dense_mlp, "_sum_dim0", eager_sum),
            patch.object(
                dense_mlp, "symm_mem_all_reduce", side_effect=lambda v, group: v * 4
            ) as reduction,
        ):
            actual = module(x, gamma)
        torch.testing.assert_close(actual, batch_reference(x, gamma, shards) * 4)
        self.assertIs(reduction.call_args.args[1], execution)


if __name__ == "__main__":
    unittest.main()
