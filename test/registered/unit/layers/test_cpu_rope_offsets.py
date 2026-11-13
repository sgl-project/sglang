"""CPU RoPE delegation must apply each position offset exactly once."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.rotary_embedding import base, rope_variant
from sglang.srt.layers.rotary_embedding.base import (
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def reference_rotation(x, rope, indices):
    """Build each rotated coordinate independently from the selected cache row."""
    cache = rope.cos_sin_cache[indices].to(x.dtype)
    cos, sin = cache.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    half = rope.rotary_dim // 2
    if rope.is_neox_style:
        left, right = x[..., :half], x[..., half : rope.rotary_dim]
    else:
        left, right = x[..., : rope.rotary_dim : 2], x[..., 1 : rope.rotary_dim : 2]
    rotated_left = left * cos - right * sin
    rotated_right = right * cos + left * sin
    if rope.is_neox_style:
        rotated = torch.cat((rotated_left, rotated_right), dim=-1)
    else:
        rotated = torch.stack((rotated_left, rotated_right), dim=-1).flatten(-2)
    return torch.cat((rotated, x[..., rope.rotary_dim :]), dim=-1)


def tensors(dtype, tokens=4, heads=3, head_size=8):
    query = torch.linspace(-1.3, 1.7, tokens * heads * head_size, dtype=dtype)
    key = torch.linspace(1.1, -0.9, tokens * head_size, dtype=dtype)
    return query.reshape(tokens, heads, head_size), key.reshape(tokens, 1, head_size)


class TestCpuRopeOffsets(unittest.TestCase):
    def setUp(self):
        self.backends = ExitStack()
        self.addCleanup(self.backends.close)
        # Select the genuine PyTorch CPU fallback, even on an AMX-capable host.
        # No operator or tensor implementation is replaced.
        self.backends.enter_context(patch.object(base, "_is_cpu_amx_available", False))
        self.backends.enter_context(
            patch.object(rope_variant, "_is_cpu_amx_available", False)
        )

    def test_base_vector_offsets_match_selected_rotations(self):
        for neox in (False, True):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(neox=neox, dtype=dtype):
                    rope = RotaryEmbedding(8, 4, 64, 10000, neox, dtype)
                    positions = torch.tensor([1, 4, 7, 15])
                    offsets = torch.tensor([2, 1, 0, 3])
                    query, key = tensors(dtype)
                    before = [
                        tensor.clone() for tensor in (query, key, positions, offsets)
                    ]
                    actual = rope.forward_cpu(positions, query, key, offsets)
                    for result, value in zip(actual, (query, key)):
                        expected = reference_rotation(value, rope, positions + offsets)
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    for tensor, original in zip(
                        (query, key, positions, offsets), before
                    ):
                        torch.testing.assert_close(tensor, original, rtol=0, atol=0)

    def test_valid_last_cache_position_with_scalar_offset(self):
        rope = RotaryEmbedding(8, 8, 64, 10000, True, torch.float32)
        positions = torch.tensor([0, 31])
        offset = torch.tensor(32)
        query, key = tensors(torch.float32, tokens=2)
        actual = rope.forward_cpu(positions, query, key, offset)
        for result, value in zip(actual, (query, key)):
            torch.testing.assert_close(
                result,
                reference_rotation(value, rope, positions + offset),
                rtol=0,
                atol=0,
            )

    def test_linear_scaling_cache_offsets(self):
        for neox in (False, True):
            with self.subTest(neox=neox):
                rope = LinearScalingRotaryEmbedding(
                    8, 8, 32, 10000, neox, [1.0, 2.0], torch.float32
                )
                positions = torch.tensor([0, 7, 31, 63])
                offset = torch.tensor(rope.scaling_factor_to_offset[2.0])
                query, key = tensors(torch.float32)
                actual = rope.forward_cpu(positions, query, key, offset)
                for result, value in zip(actual, (query, key)):
                    torch.testing.assert_close(
                        result,
                        reference_rotation(value, rope, positions + offset),
                        rtol=0,
                        atol=0,
                    )

    def test_deepseek_offsets_preserve_native_layouts(self):
        for neox in (False, True):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(neox=neox, dtype=dtype):
                    rope = DeepseekScalingRotaryEmbedding(
                        8, 4, 64, 10000, neox, 2.0, dtype, device="cpu"
                    )
                    positions = torch.tensor([0, 2, 4, 3, 1, 5])
                    offsets = torch.tensor([2, 1, 0, 1, 3, 4])
                    query, key = tensors(dtype, tokens=6)
                    # The NeoX DeepSeek implementation documents batched positions.
                    if neox:
                        positions = positions.reshape(2, 3)
                        offsets = offsets.reshape(2, 3)
                        query = query.reshape(2, 3, 3, 8)
                        key = key.reshape(2, 3, 1, 8)
                    actual = rope.forward_cpu(positions, query, key, offsets)
                    for result, value in zip(actual, (query, key)):
                        self.assertEqual(result.shape, value.shape)
                        torch.testing.assert_close(
                            result,
                            reference_rotation(value, rope, positions + offsets),
                            rtol=0,
                            atol=0,
                        )

    def test_no_offset_and_zero_offset_controls(self):
        for neox in (False, True):
            rope = RotaryEmbedding(8, 4, 64, 10000, neox, torch.float32)
            positions = torch.tensor([1, 4, 7, 15])
            query, key = tensors(torch.float32)
            for offsets in (None, torch.zeros_like(positions)):
                with self.subTest(neox=neox, offsets=offsets):
                    actual = rope.forward_cpu(positions, query, key, offsets)
                    for result, value in zip(actual, (query, key)):
                        torch.testing.assert_close(
                            result,
                            reference_rotation(value, rope, positions),
                            rtol=0,
                            atol=0,
                        )

    def test_attention_outputs_and_input_gradients(self):
        rope = RotaryEmbedding(8, 4, 64, 10000, True, torch.float64)
        positions = torch.tensor([1, 4, 7, 15])
        offsets = torch.tensor([1, 5, 0, 3])
        query, key = tensors(torch.float64, heads=1)
        query.requires_grad_()
        key.requires_grad_()
        reference_query = query.detach().clone().requires_grad_()
        reference_key = key.detach().clone().requires_grad_()
        actual_q, actual_k = rope.forward_cpu(positions, query, key, offsets)
        expected_q = reference_rotation(reference_query, rope, positions + offsets)
        expected_k = reference_rotation(reference_key, rope, positions + offsets)
        values = torch.arange(12, dtype=torch.float64).reshape(1, 4, 3) / 12
        actual = F.scaled_dot_product_attention(
            actual_q.transpose(0, 1), actual_k.transpose(0, 1), values
        )
        expected = F.scaled_dot_product_attention(
            expected_q.transpose(0, 1), expected_k.transpose(0, 1), values
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(query.grad, reference_query.grad, rtol=0, atol=0)
        torch.testing.assert_close(key.grad, reference_key.grad, rtol=0, atol=0)

    def test_unsupported_fused_cache_argument_still_rejected(self):
        rope = RotaryEmbedding(8, 4, 64, 10000, True, torch.float32)
        query, key = tensors(torch.float32)
        with self.assertRaisesRegex(AssertionError, "not supported for cpu"):
            rope.forward_cpu(
                torch.arange(4), query, key, fused_set_kv_buffer_arg=object()
            )


if __name__ == "__main__":
    unittest.main()
