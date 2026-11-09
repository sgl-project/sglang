"""Correctness and fallback coverage for DeepSeek-V4.1 vision RoPE."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_v41_vit import apply_rotary, apply_vision_rotary
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestDeepseekV41VisionRoPE(CustomTestCase):
    def _compare(self, q, k, cos, sin):
        q_before, k_before = q.clone(), k.clone()
        actual_q, actual_k = apply_vision_rotary(q, k, (cos, sin), None)
        expected_q = apply_rotary(q, cos, sin)
        expected_k = apply_rotary(k, cos, sin)
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(q, q_before, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(k, k_before, rtol=0, atol=0, equal_nan=True)
        return actual_q, actual_k

    def test_strided_qk_and_tail_heads(self):
        torch.manual_seed(42)
        for dtype in (torch.bfloat16, torch.float16):
            for tokens, heads in (
                (1, 1),
                (37, 7),
                (64, 8),
                (256, 15),
                (1521, 16),
                (128, 17),
            ):
                with self.subTest(dtype=dtype, tokens=tokens, heads=heads):
                    qkv = torch.randn(
                        tokens, 3 * heads * 64, device="cuda", dtype=dtype
                    )
                    q = qkv[:, : heads * 64].view(tokens, heads, 64)
                    k = qkv[:, heads * 64 : 2 * heads * 64].view(tokens, heads, 64)
                    angles = torch.randn(tokens, 1, 32, device="cuda")
                    self._compare(q, k, angles.cos(), angles.sin())

    def test_all_16bit_values(self):
        for dtype in (torch.bfloat16, torch.float16):
            raw = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
            q = raw.view(dtype).reshape(64, 16, 64)
            k = q.flip(0)
            for seed in (0, 1, 2, 3):
                with self.subTest(dtype=dtype, seed=seed):
                    gen = torch.Generator(device="cuda").manual_seed(seed)
                    angles = (
                        torch.rand(64, 1, 32, device="cuda", generator=gen) * 2000
                        - 1000
                    )
                    cos, sin = angles.cos(), angles.sin()
                    actual = self._compare(q, k, cos, sin)
                    expected = (apply_rotary(q, cos, sin), apply_rotary(k, cos, sin))
                    for got, want in zip(actual, expected):
                        finite = torch.isfinite(want) & torch.isfinite(got)
                        self.assertTrue(
                            torch.equal(
                                got.view(torch.int16)[finite],
                                want.view(torch.int16)[finite],
                            )
                        )

    def test_cuda_graph_replay(self):
        q = torch.randn(73, 16, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        angles = torch.randn(73, 1, 32, device="cuda")
        cos, sin = angles.cos(), angles.sin()
        for _ in range(3):
            apply_vision_rotary(q, k, (cos, sin), None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = apply_vision_rotary(q, k, (cos, sin), None)
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn_like(k))
        graph.replay()
        expected = (apply_rotary(q, cos, sin), apply_rotary(k, cos, sin))
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_rocm_uses_native_fallback(self):
        q = torch.randn(17, 16, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        angles = torch.randn(17, 1, 32, device="cuda")
        cos, sin = angles.cos(), angles.sin()
        with (
            patch("torch.version.hip", "mock_hip"),
            patch(
                "sglang.srt.models.deepseek_v41_vit.apply_rotary", wraps=apply_rotary
            ) as native,
        ):
            actual = apply_vision_rotary(q, k, (cos, sin), None)
        self.assertEqual(native.call_count, 2)
        torch.testing.assert_close(actual[0], apply_rotary(q, cos, sin), rtol=0, atol=0)
        torch.testing.assert_close(actual[1], apply_rotary(k, cos, sin), rtol=0, atol=0)

    def test_fallback_for_autograd_noncontiguous_and_empty(self):
        q = torch.randn(3, 2, 64, device="cuda", requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        backing = torch.randn(3, 1, 64, device="cuda", requires_grad=True)
        cos = backing[..., ::2]
        sin = backing[..., 1::2]
        actual_q, actual_k = self._compare(q, k, cos, sin)
        (actual_q.sum() + actual_k.sum()).backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(backing.grad)

        empty = torch.empty(0, 2, 64, device="cuda", dtype=torch.bfloat16)
        empty_positions = torch.empty(0, 1, 32, device="cuda", dtype=torch.float32)
        actual = apply_vision_rotary(
            empty, empty, (empty_positions, empty_positions), None
        )
        self.assertEqual(actual[0].shape, empty.shape)
        self.assertEqual(actual[1].shape, empty.shape)


if __name__ == "__main__":
    unittest.main()
