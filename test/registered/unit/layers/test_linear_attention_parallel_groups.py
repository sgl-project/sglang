"""Linear attention projections retain checkpoint layout across reload scopes."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.runtime_context import SpawnRanks, get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cpu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def build_linear_attention(
    model,
    n_groups=8,
    reduce_results=None,
    width=32,
    head_dim=8,
    quant_config=None,
    heads=8,
    device="cpu",
):
    if model == "mamba":
        from sglang.srt.configs.mamba_utils import Mamba2StateShape
        from sglang.srt.layers.attention.mamba.mamba import MambaMixer2

        size = (
            get_parallel().attn_tp_size
            if get_parallel().attn_dp_size > 1
            else get_parallel().tp_size
        )
        shape = Mamba2StateShape.create(
            tp_world_size=size,
            intermediate_size=heads * head_dim,
            n_groups=n_groups,
            num_heads=heads,
            head_dim=head_dim,
            state_size=head_dim,
            conv_kernel=3,
        )
        module = MambaMixer2(
            SimpleNamespace(shape=shape),
            width,
            use_conv_bias=True,
            use_bias=True,
            n_groups=n_groups,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
        names = ("conv1d", "in_proj", "out_proj")
    else:
        config = SimpleNamespace(
            hidden_size=width,
            linear_num_value_heads=heads,
            linear_num_key_heads=heads,
            linear_num_value_heads_cpu=heads,
            linear_num_key_heads_cpu=heads,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
            linear_conv_kernel_dim=3,
            hidden_act="silu",
            output_gate_type=None,
            rms_norm_eps=1e-6,
            torch_dtype=torch.get_default_dtype(),
        )
        if model in ("qwen_next", "gigachat"):
            from sglang.srt.models.qwen3_next import Qwen3GatedDeltaNet

            constructor = Qwen3GatedDeltaNet
            if model == "gigachat":
                from sglang.srt.models.gigachat35 import GigaChat35GatedDeltaNet

                constructor = GigaChat35GatedDeltaNet
        else:
            from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

            constructor = Qwen3_5GatedDeltaNet
        with patch(
            "torch.get_device_module",
            return_value=SimpleNamespace(current_device=lambda: device),
        ):
            module = constructor(config, 0, quant_config=quant_config)
        names = ("conv1d", "in_proj_qkvz", "in_proj_ba", "out_proj")
    return module, names


def checkpoint_values(rows, columns, layer, offset=0):
    return (
        (torch.arange(rows * columns, device=layer.weight.device) + offset) % 23 - 11
    ).reshape(rows, columns).to(layer.weight.dtype) / 128


def load_projection(module, name, n_groups=8, checkpoint="fused"):
    """Return an independently sliced checkpoint reference, including copied groups."""
    layer = getattr(module, name)
    rank, size = layer.tp_rank, layer.tp_size
    row = isinstance(layer, RowParallelLinear)
    mamba = hasattr(module, "intermediate_size")
    duplicate = mamba and n_groups == 1 and size > 1
    if name == "conv1d":
        sizes = (
            [
                module.intermediate_size,
                n_groups * module.ssm_state_size,
                n_groups * module.ssm_state_size,
            ]
            if mamba
            else [module.key_dim, module.key_dim, module.value_dim]
        )
    elif mamba and name == "in_proj":
        sizes = [
            module.intermediate_size,
            module.intermediate_size,
            n_groups * module.ssm_state_size,
            n_groups * module.ssm_state_size,
            module.num_heads,
        ]
    else:
        sizes = getattr(layer, "output_sizes", [layer.output_size])
    weight = checkpoint_values(sum(sizes), layer.input_size, layer)
    pieces = weight.split(sizes)
    if row:
        shard = weight.chunk(size, dim=1)[rank]
    elif duplicate:
        copied = (1, 2) if name == "conv1d" else (2, 3)
        shard = torch.cat(
            [
                piece if i in copied else piece.chunk(size)[rank]
                for i, piece in enumerate(pieces)
            ]
        )
    elif (
        module.__class__.__name__ in ("Qwen3GatedDeltaNet", "GigaChat35GatedDeltaNet")
        and name.startswith("in_proj")
        and checkpoint == "fused"
    ):
        shard = weight.chunk(size)[rank]
    else:
        shard = torch.cat([piece.chunk(size)[rank] for piece in pieces])
    bias_shard = None
    with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
        if name.startswith("in_proj") and not mamba and checkpoint != "fused":
            if checkpoint == "tuple":
                indices = tuple(range(len(sizes) - 1))
                layer.weight.weight_loader(
                    layer.weight, torch.cat(pieces[:-1]), indices
                )
                layer.weight.weight_loader(layer.weight, pieces[-1], len(sizes) - 1)
            else:
                for i, piece in enumerate(pieces):
                    layer.weight.weight_loader(layer.weight, piece, i)
        else:
            loaded = weight.unsqueeze(1) if name == "conv1d" else weight
            layer.weight.weight_loader(layer.weight, loaded)
        if layer.bias is not None:
            rows = sum(sizes) if name == "conv1d" else layer.output_size
            bias = checkpoint_values(rows, 1, layer, 3).flatten()
            layer.bias.weight_loader(layer.bias, bias)
            if row:
                bias_shard = bias
            elif duplicate and name == "conv1d":
                bias_shard = torch.cat(
                    [
                        p if i in (1, 2) else p.chunk(size)[rank]
                        for i, p in enumerate(bias.split(sizes))
                    ]
                )
            elif duplicate:
                bias_shard = bias.chunk(size)[rank]
            else:
                bias_shard = torch.cat([p.chunk(size)[rank] for p in bias.split(sizes)])
    return shard.unsqueeze(1) if name == "conv1d" else shard, bias_shard


class TestLinearAttentionParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_checkpoint_reload_projection_math_and_row_policy(self):
        # The existing CPU bias-padding loader cannot load duplicated Mamba groups.
        variants = [
            (model, 8, None) for model in ("qwen_next", "qwen35", "gigachat")
        ] + [
            ("mamba", groups, reduce)
            for groups in ((8,) if is_cpu() else (1, 8))
            for reduce in (None, False, True)
        ]
        for dp_size in (1, 2):
            for rank in (0, 3):
                server = ServerArgs(
                    model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp_size
                )
                reset_context()
                publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
                initialize_dp_attention_flags(server)
                size = 4 // dp_size
                tp, attn = Mock(), Mock()
                tp.all_reduce.side_effect = lambda x: x * 4
                attn.all_reduce.side_effect = lambda x: x * size
                for model, groups, reduce in variants:
                    with self.subTest(
                        dp=dp_size, rank=rank, model=model, groups=groups, reduce=reduce
                    ):
                        module, names = build_linear_attention(model, groups, reduce)
                        for name in names:
                            layer = getattr(module, name)
                            modes = (
                                (
                                    ("fused", "split", "tuple")
                                    if model == "qwen35"
                                    else ("fused", "split")
                                )
                                if model != "mamba" and name.startswith("in_proj")
                                else ("fused",)
                            )
                            for checkpoint in modes:
                                shard, bias = load_projection(
                                    module, name, groups, checkpoint
                                )
                                torch.testing.assert_close(layer.weight, shard)
                                if bias is not None:
                                    torch.testing.assert_close(layer.bias, bias)
                                self.assertEqual(
                                    (layer.tp_rank, layer.tp_size), (rank % size, size)
                                )
                                row = isinstance(layer, RowParallelLinear)
                                if name == "conv1d":
                                    inputs = checkpoint_values(
                                        2 * shard.shape[0], 5, layer, 7
                                    ).reshape(2, shard.shape[0], 5)
                                    actual = F.conv1d(
                                        inputs,
                                        layer.weight,
                                        layer.bias,
                                        groups=shard.shape[0],
                                    )
                                    expected = (
                                        inputs.unfold(-1, 3, 1)
                                        * shard.squeeze(1)[None, :, None, :]
                                    ).sum(-1)
                                    if bias is not None:
                                        expected += bias[None, :, None]
                                    torch.testing.assert_close(actual, expected)
                                    continue
                                inputs = checkpoint_values(2, shard.shape[1], layer, 7)
                                used_bias = (
                                    bias if not row or layer.tp_rank == 0 else None
                                )
                                expected = F.linear(inputs, shard, used_bias)
                                if row and layer.reduce_results:
                                    expected *= (
                                        size if layer.use_dp_attention_reduce else 4
                                    )
                                tp.all_reduce.reset_mock()
                                attn.all_reduce.reset_mock()
                                with (
                                    parallel_scope(tp_group=tp, attn_tp_group=attn),
                                    patch(
                                        "sglang.srt.layers.linear.is_allocation_symmetric",
                                        return_value=True,
                                    ),
                                    patch(
                                        "sglang.srt.layers.linear.use_symmetric_memory",
                                        return_value=nullcontext(),
                                    ) as allocator,
                                ):
                                    actual = layer(inputs)[0]
                                    torch.testing.assert_close(actual, expected)
                                if row:
                                    group = (
                                        attn if layer.use_dp_attention_reduce else tp
                                    )
                                    expected_kwargs = (
                                        {}
                                        if layer.use_dp_attention_reduce
                                        else {"disabled": False}
                                    )
                                    allocator.assert_called_once_with(
                                        group, **expected_kwargs
                                    )
                                    self.assertEqual(
                                        tp.all_reduce.call_count
                                        + attn.all_reduce.call_count,
                                        int(layer.reduce_results),
                                    )
                                    self.assertEqual(
                                        layer.reduce_results,
                                        (dp_size == 1 if reduce is None else reduce)
                                        if model == "mamba"
                                        else False,
                                    )
                                    self.assertEqual(
                                        layer.use_dp_attention_reduce,
                                        model == "mamba"
                                        and layer.reduce_results
                                        and dp_size > 1,
                                    )
                                else:
                                    allocator.assert_not_called()

    def test_projection_helpers_default_to_full_tp(self):
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for model in ("qwen_next", "qwen35"):
            module, _ = build_linear_attention(model)
            projections = [module.create_qkvz_proj(32, 64, 64, None, "")]
            if model == "qwen35":
                projections.append(module.create_ba_proj(32, 8, None, ""))
            for layer in projections:
                self.assertEqual((layer.tp_rank, layer.tp_size), (3, 4))


if __name__ == "__main__":
    unittest.main()
