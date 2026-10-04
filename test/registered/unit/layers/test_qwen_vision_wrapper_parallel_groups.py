"""Qwen vision wrappers and generation helpers retain their placement policies."""

import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def group_kwargs(cls, group):
    if group is None:
        return {}
    if "parallel_group" in inspect.signature(cls).parameters:
        return dict(parallel_group=group)
    p = get_parallel()
    rank, size = (
        (0, 1)
        if group == "replicated"
        else (p.tp_rank, p.tp_size)
        if group == "tp"
        else (p.attn_tp_rank, p.attn_tp_size)
    )
    return dict(tp_rank=rank, tp_size=size)


def build_qwen(model, group=None, *, replicated=False, width=32, quant_config=None):
    options = dict(quant_config=quant_config, use_data_parallel=replicated)
    if model.startswith("qwen25"):
        from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLMLP

        module = Qwen2_5_VLMLP(
            width,
            2 * width,
            bias=True,
            fuse_gate_up=model == "qwen25_fused",
            **options,
            **group_kwargs(Qwen2_5_VLMLP, group),
        )
        layers = (
            (module.gate_up_proj, module.down_proj)
            if module.fuse_gate_up
            else (module.gate_proj, module.up_proj, module.down_proj)
        )
    elif model == "qwen3_mlp":
        from sglang.srt.models.qwen3_vl import Qwen3_VisionMLP

        module = Qwen3_VisionMLP(
            width,
            2 * width,
            bias=True,
            **options,
            **group_kwargs(Qwen3_VisionMLP, group),
        )
        layers = (module.linear_fc1, module.linear_fc2)
    elif model == "qwen3_merger":
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionPatchMerger

        module = Qwen3VLMoeVisionPatchMerger(
            width,
            width,
            2 * width,
            spatial_merge_size=1,
            **options,
            **group_kwargs(Qwen3VLMoeVisionPatchMerger, group),
        )
        layers = (module.linear_fc1, module.linear_fc2)
    else:
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            global_force_attn_backend_context_manager,
        )
        from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_vision import (
            QwenVLVisionAttention,
        )
        from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

        with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
            module = QwenVLVisionAttention(
                SimpleNamespace(hidden_size=width, num_heads=8),
                prefix="visual",
                model_name="Qwen",
                quant_config=quant_config,
            )
        layers = (module.qkv_proj, module.proj)
    return module, layers


def load_projection(layer):
    """Load known full weights under rank zero, returning the expected owned shard."""
    rank, size = getattr(layer, "tp_rank", 0), getattr(layer, "tp_size", 1)
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

    with get_parallel().override(tp_rank=0, attn_dp_rank=0, attn_tp_rank=0):
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


class TestQwenVisionWrapperGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_native_projection_math_loaders_and_flags(self):
        for dp in (1, 2):
            reset_context()
            server = ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=4,
                attn_dp_size=dp,
                mm_attention_backend="sdpa",
            )
            publish(server, role="test", ranks=SpawnRanks(world_rank=3))
            initialize_dp_attention_flags(server)
            tp, attn = Mock(), Mock()
            tp.all_reduce.side_effect = lambda x: x * 2
            attn.all_reduce.side_effect = lambda x: x * 3
            for model in (
                "qwen25_fused",
                "qwen25_split",
                "qwen3_mlp",
                "qwen3_merger",
                "mmdg_attention",
            ):
                for group, replicated in (
                    ((None, False),)
                    if model == "mmdg_attention"
                    else (
                        (None, False),
                        (None, True),
                        ("tp", False),
                        ("attn_tp", False),
                        ("replicated", False),
                    )
                ):
                    with self.subTest(
                        dp=dp, model=model, group=group, replicated=replicated
                    ):
                        module, layers = build_qwen(model, group, replicated=replicated)
                        selected = (
                            "replicated"
                            if replicated
                            else group
                            or ("attn_tp" if model.startswith("qwen3") else "tp")
                        )
                        expected_rank, expected_size = (
                            (0, 1)
                            if selected == "replicated"
                            else (3, 4)
                            if selected == "tp"
                            else (3 % (4 // dp), 4 // dp)
                        )
                        for index, layer in enumerate(layers):
                            weight, bias = load_projection(layer)
                            row = isinstance(layer, RowParallelLinear)
                            inputs = (
                                torch.arange(2 * weight.shape[1]).reshape(2, -1).float()
                                / 64
                            )
                            expected = F.linear(
                                inputs,
                                weight,
                                bias if not row or expected_rank == 0 else None,
                            )
                            dp_reduce = row and layer.use_dp_attention_reduce
                            if row and expected_size > 1:
                                expected *= 3 if dp_reduce else 2
                            tp.all_reduce.reset_mock()
                            attn.all_reduce.reset_mock()
                            with (
                                get_parallel().override(
                                    tp_group=tp, attn_tp_group=attn
                                ),
                                patch(
                                    "sglang.srt.layers.linear.is_allocation_symmetric",
                                    return_value=False,
                                ),
                                patch(
                                    "sglang.srt.layers.linear.use_symmetric_memory",
                                    return_value=nullcontext(),
                                ) as allocator,
                            ):
                                actual = layer(inputs)[0]
                                torch.testing.assert_close(actual, expected)
                                if row:
                                    allocator.assert_called_once_with(
                                        attn
                                    ) if dp_reduce else allocator.assert_called_once_with(
                                        tp, disabled=True
                                    )
                                else:
                                    allocator.assert_not_called()
                                self.assertEqual(
                                    tp.all_reduce.call_count,
                                    int(row and expected_size > 1 and not dp_reduce),
                                )
                                self.assertEqual(
                                    attn.all_reduce.call_count,
                                    int(row and expected_size > 1 and dp_reduce),
                                )
                            self.assertEqual(
                                (
                                    getattr(layer, "tp_rank", 0),
                                    getattr(layer, "tp_size", 1),
                                ),
                                (expected_rank, expected_size),
                            )
                        if model == "qwen25_split":
                            self.assertEqual(
                                isinstance(module.down_proj, ReplicatedLinear),
                                expected_size == 1,
                            )

    def test_generation_helpers_match_initialized_scope(self):
        import sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl as generation

        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for initialized in (False, True):
            with (
                patch.object(
                    generation,
                    "model_parallel_is_initialized",
                    return_value=initialized,
                ),
                patch.object(generation, "get_tp_world_size", return_value=4),
            ):
                # Parent-only helper reads the same rank as the published bridge scope.
                with patch.object(
                    generation, "get_tp_rank", return_value=3, create=True
                ):
                    for tensor_parallel in (False, True):
                        for maker in (
                            generation._make_column_linear,
                            generation._make_row_linear,
                        ):
                            layer = maker(
                                8, 8, bias=True, use_tensor_parallel=tensor_parallel
                            )
                            expected = (
                                (3, 4) if initialized and tensor_parallel else (0, 1)
                            )
                            self.assertEqual(
                                (
                                    getattr(layer, "tp_rank", 0),
                                    getattr(layer, "tp_size", 1),
                                ),
                                expected,
                            )
                            self.assertEqual(
                                isinstance(layer, ReplicatedLinear), not tensor_parallel
                            )

    def test_wrapper_conflict_guard_and_disabled_merger(self):
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_cp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for model in ("qwen25_fused", "qwen3_mlp", "qwen3_merger"):
            with self.assertRaisesRegex(
                ValueError, "cannot be combined with data parallel"
            ):
                build_qwen(model, "replicated", replicated=True)
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionPatchMerger

        reset_context()
        merger = Qwen3VLMoeVisionPatchMerger(
            32,
            32,
            64,
            disable_merger_proj=True,
            **group_kwargs(Qwen3VLMoeVisionPatchMerger, "replicated"),
        )
        self.assertFalse(hasattr(merger, "linear_fc1"))
        self.assertFalse(hasattr(merger, "tp_size"))


if __name__ == "__main__":
    unittest.main()
