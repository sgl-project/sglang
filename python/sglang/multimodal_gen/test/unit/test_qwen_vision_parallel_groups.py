"""Qwen generation projections tested in the diffusion dependency environment."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.linear import QKVParallelLinear, ReplicatedLinear
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.parallel_groups import parallel_scope, publish


class TestQwenGenerationParallelGroups(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_generation_helpers_match_initialized_scope(self):
        import sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl as generation

        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for initialized in (False, True):
            with patch.object(
                generation,
                "model_parallel_is_initialized",
                return_value=initialized,
            ):
                for tensor_parallel in (False, True):
                    for maker in (
                        generation._make_column_linear,
                        generation._make_row_linear,
                    ):
                        layer = maker(
                            8, 8, bias=True, use_tensor_parallel=tensor_parallel
                        )
                        expected = (3, 4) if initialized and tensor_parallel else (0, 1)
                        self.assertEqual((layer.tp_rank, layer.tp_size), expected)
                        self.assertEqual(
                            isinstance(layer, ReplicatedLinear), not tensor_parallel
                        )

    def test_native_vision_projection_math_and_reload(self):
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            global_force_attn_backend_context_manager,
        )
        from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_vision import (
            QwenVLVisionAttention,
        )
        from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

        for dp in (1, 2):
            with self.subTest(dp=dp):
                reset_context()
                server = ServerArgs(
                    model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp
                )
                publish(server, role="test", ranks=SpawnRanks(world_rank=3))
                initialize_dp_attention_flags(server)
                with global_force_attn_backend_context_manager(
                    AttentionBackendEnum.TORCH_SDPA
                ):
                    module = QwenVLVisionAttention(
                        SimpleNamespace(hidden_size=32, num_heads=8),
                        prefix="visual",
                        model_name="Qwen",
                    )
                tp = Mock(all_reduce=Mock(side_effect=lambda x: x * 2))
                attn = Mock(all_reduce=Mock(side_effect=AssertionError("wrong group")))
                for layer in (module.qkv_proj, module.proj):
                    rank, size = (
                        getattr(layer, "tp_rank", 0),
                        getattr(layer, "tp_size", 1),
                    )
                    self.assertEqual((rank, size), (3, 4))
                    full = (
                        torch.arange(32 * 32, dtype=layer.weight.dtype).reshape(32, 32)
                        / 1024
                    )
                    bias = torch.arange(32, dtype=layer.weight.dtype) / 64
                    qkv = isinstance(layer, QKVParallelLinear)
                    with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
                        for shard in ("q", "k", "v") if qkv else (None,):
                            args = (shard,) if qkv else ()
                            layer.weight.weight_loader(layer.weight, full, *args)
                            layer.bias.weight_loader(layer.bias, bias, *args)
                    weight = (
                        torch.cat([full.chunk(size)[rank]] * 3)
                        if qkv
                        else full.chunk(size, dim=1)[rank]
                    )
                    expected_bias = (
                        torch.cat([bias.chunk(size)[rank]] * 3) if qkv else None
                    )
                    inputs = (
                        torch.arange(2 * weight.shape[1], dtype=weight.dtype).reshape(
                            2, -1
                        )
                        / 64
                    )
                    expected = F.linear(inputs, weight, expected_bias) * (
                        1 if qkv else 2
                    )
                    with (
                        parallel_scope(tp_group=tp, attn_tp_group=attn),
                        patch(
                            "sglang.srt.layers.linear.is_allocation_symmetric",
                            return_value=False,
                        ),
                        patch(
                            "sglang.srt.layers.linear.use_symmetric_memory",
                            return_value=nullcontext(),
                        ),
                    ):
                        torch.testing.assert_close(layer(inputs)[0], expected)
                    torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
                attn.all_reduce.assert_not_called()
                tp.all_reduce.assert_called_once()
