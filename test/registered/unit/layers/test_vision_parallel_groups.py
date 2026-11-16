"""Vision projections retain their layout and execution policy across reload scopes."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.runtime_context import (
    SpawnRanks,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, stage="weekly", runner_config="cpu")

VISION_MODELS = (
    "vision_packed",
    "vision_column",
    "vision_dummy",
    "vision_gqa",
    "vision_legacy",
    "siglip",
    "glm4v_mlp",
    "glm4v_merger",
    "internvl_mlp",
    "qwen_merger",
    "minimax_projector",
    "minimax_merger",
    "minimax_encoder",
    "moonvit",
    "step3",
)


def build_vision(model, use_data_parallel=False, width=32, quant_config=None):
    config = SimpleNamespace(
        hidden_size=width,
        intermediate_size=2 * width,
        num_attention_heads=8,
        layer_norm_eps=1e-6,
        hidden_act="gelu",
    )
    options = dict(quant_config=quant_config, use_data_parallel=use_data_parallel)
    if model.startswith("vision_"):
        from sglang.srt.layers.attention.vision import VisionAttention

        module = VisionAttention(
            width,
            8,
            width,
            use_qkv_parallel=model != "vision_column",
            num_dummy_heads=4 if model == "vision_dummy" else 0,
            num_kv_heads=4 if model == "vision_gqa" else 8,
            use_dp_attention_reduce=model != "vision_legacy",
            allow_tp_reduce_mismatch=model == "vision_legacy",
            qkv_backend="sdpa",
            **options,
        )
        layers = (module.qkv_proj, module.proj)
    elif model in ("glm4v_mlp", "glm4v_merger"):
        from sglang.srt.models.glm4v import Glm4vPatchMerger, Glm4vVisionMLP

        constructor = Glm4vVisionMLP if model == "glm4v_mlp" else Glm4vPatchMerger
        module = constructor(width, 2 * width, bias=True, **options)
        layers = (module.gate_up_proj, module.down_proj)
    elif model == "internvl_mlp":
        from sglang.srt.models.internvl import InternMLP

        module = InternMLP(config, use_data_parallel=use_data_parallel)
        layers = (module.fc1, module.fc2)
    elif model == "siglip":
        from sglang.srt.models.siglip import SiglipMLP

        module = SiglipMLP(config, **options)
        layers = (module.fc1, module.fc2)
    elif model == "qwen_merger":
        from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionPatchMerger

        module = Qwen2_5_VisionPatchMerger(
            width,
            width,
            2 * width,
            spatial_merge_size=1,
            force_native_norm=True,
            **options,
        )
        layers = (module.mlp[0], module.mlp[2])
    elif model == "moonvit":
        from sglang.srt.models.kimi_vl_moonvit import MLP2

        module = MLP2(
            [width, 2 * width, width],
            torch.nn.GELU(),
            use_tensor_parallel=True,
            **options,
        )
        layers = (module.fc0, module.fc1)
    elif model == "step3":
        from sglang.srt.models.step3_vl import Step3VisionMLP

        module = Step3VisionMLP(width, 2 * width, quant_config=quant_config)
        layers = (module.fc1, module.fc2)
    else:
        from sglang.srt.models.minimax_vl_common import (
            CLIPEncoderLayer,
            MiniMaxVLMultiModalProjector,
            MiniMaxVLPatchMerger,
        )

        if model == "minimax_projector":
            module = MiniMaxVLMultiModalProjector(
                width, width, "gelu", True, projector_hidden_size=2 * width, **options
            )
            layers = (module.linear_1, module.linear_2)
        elif model == "minimax_merger":
            module = MiniMaxVLPatchMerger(
                1, width, "gelu", True, projector_hidden_size=2 * width, **options
            )
            layers = (module.linear_1, module.linear_2)
        else:
            module = CLIPEncoderLayer(config, **options)
            layers = (
                module.fc1,
                module.fc2,
                module.self_attn.qkv_proj,
                module.self_attn.proj,
            )
    return module, layers


def load_projection(layer):
    """Load known full weights under rank zero, returning the expected owned shard."""
    rank, size = rank_size(layer)[0], rank_size(layer)[1]
    row = isinstance(layer, RowParallelLinear)
    dtype, device = layer.weight.dtype, layer.weight.device

    def values(rows, columns, offset):
        return (
            (
                torch.arange(rows * columns, device=device).reshape(rows, columns)
                + offset
            )
            % 23
            / 128
        ).to(dtype)

    with parallel_scope(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
        if isinstance(layer, QKVParallelLinear):
            weights, biases = [], []
            for offset, shard_id in enumerate(("q", "k", "v")):
                heads = (
                    layer.total_num_heads
                    if shard_id == "q"
                    else layer.total_num_kv_heads
                )
                weight = values(heads * layer.head_size, layer.input_size, offset)
                partitions = size if shard_id == "q" else min(heads, size)
                index = rank if shard_id == "q" else rank // max(size // heads, 1)
                layer.weight.weight_loader(layer.weight, weight, shard_id)
                weights.append(weight.chunk(partitions)[index])
                if layer.bias is not None:
                    bias = values(heads * layer.head_size, 1, offset).flatten()
                    layer.bias.weight_loader(layer.bias, bias, shard_id)
                    biases.append(bias.chunk(partitions)[index])
            shard = torch.cat(weights)
            bias_shard = torch.cat(biases) if biases else None
        else:
            weight = values(layer.output_size, layer.input_size, 3)
            layer.weight.weight_loader(layer.weight, weight)
            shard = (
                weight.chunk(size, dim=1)[rank]
                if row
                else torch.cat(
                    [
                        p.chunk(size)[rank]
                        for p in weight.split(
                            getattr(layer, "output_sizes", [layer.output_size])
                        )
                    ]
                )
            )
            bias_shard = None
            if layer.bias is not None:
                bias = values(layer.output_size, 1, 4).flatten()
                layer.bias.weight_loader(layer.bias, bias)
                bias_shard = (
                    bias
                    if row
                    else torch.cat(
                        [
                            p.chunk(size)[rank]
                            for p in bias.split(
                                getattr(layer, "output_sizes", [layer.output_size])
                            )
                        ]
                    )
                )
    return shard, bias_shard


class TestVisionParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_projection_layout_reload_and_row_policy(self):
        for dp_size in (1, 2):
            publish(
                ServerArgs(
                    model_path="dummy",
                    device="cpu",
                    tp_size=4,
                    attn_dp_size=dp_size,
                    enable_dp_attention=dp_size > 1,
                    mm_attention_backend="sdpa",
                ),
                role="test",
                ranks=SpawnRanks(world_rank=3),
            )
            initialize_dp_attention_flags(
                ServerArgs(
                    model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp_size
                )
            )
            tp, attn = Mock(), Mock()
            tp.all_reduce.side_effect = lambda x: x * 4
            attn.all_reduce.side_effect = lambda x: x * (4 // dp_size)
            for model in VISION_MODELS:
                for replicated in (
                    (False,) if model in ("moonvit", "step3") else (False, True)
                ):
                    with self.subTest(dp=dp_size, model=model, replicated=replicated):
                        if dp_size == 2 and model in ("moonvit", "step3"):
                            with self.assertRaisesRegex(
                                ValueError, "shards over the attention TP group"
                            ):
                                build_vision(model, replicated)
                            continue
                        module, layers = build_vision(model, replicated)
                        group = (
                            "tp"
                            if model
                            in (
                                "siglip",
                                "qwen_merger",
                                "glm4v_mlp",
                                "glm4v_merger",
                                "internvl_mlp",
                            )
                            else "attn_tp"
                        )
                        rank, size = (
                            (0, 1)
                            if replicated
                            else (
                                (3, 4)
                                if group == "tp"
                                else (3 // dp_size, 4 // dp_size)
                            )
                        )
                        for layer in layers:
                            self.assertEqual(rank_size(layer), (rank, size))
                            shard, bias = load_projection(layer)
                            torch.testing.assert_close(layer.weight, shard)
                            if bias is not None:
                                torch.testing.assert_close(layer.bias, bias)
                            row = isinstance(layer, RowParallelLinear)
                            inputs = (
                                torch.arange(
                                    2 * shard.shape[1], dtype=shard.dtype
                                ).reshape(2, -1)
                                / 128
                            )
                            expected = F.linear(
                                inputs, shard, bias if not row or rank == 0 else None
                            )
                            if row and size > 1:
                                expected *= (
                                    (4 // dp_size)
                                    if layer.use_dp_attention_reduce
                                    else 4
                                )
                            tp.all_reduce.reset_mock()
                            attn.all_reduce.reset_mock()
                            with (
                                parallel_scope(tp_group=tp, attn_tp_group=attn),
                                patch(
                                    "sglang.srt.layers.linear.is_allocation_symmetric",
                                    return_value=False,
                                ),
                                patch(
                                    "sglang.srt.layers.linear.use_symmetric_memory",
                                    return_value=nullcontext(),
                                ) as allocator,
                            ):
                                torch.testing.assert_close(layer(inputs)[0], expected)
                            if row:
                                self.assertTrue(layer.reduce_results)
                                reduction = (
                                    attn if layer.use_dp_attention_reduce else tp
                                )
                                if layer.use_dp_attention_reduce:
                                    allocator.assert_called_once_with(reduction)
                                else:
                                    allocator.assert_called_once_with(
                                        reduction, disabled=True
                                    )
                                self.assertEqual(
                                    reduction.all_reduce.call_count, int(size > 1)
                                )
                                self.assertEqual(
                                    (
                                        tp if reduction is attn else attn
                                    ).all_reduce.call_count,
                                    0,
                                )
                            else:
                                allocator.assert_not_called()
                            self.assertEqual(rank_size(layer), (rank, size))
                        if hasattr(module, "tp_rank"):
                            self.assertEqual(
                                (module.tp_rank, module.tp_size), (rank, size)
                            )
            reset_context()


if __name__ == "__main__":
    unittest.main()
