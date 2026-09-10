"""YaRN cache growth must continue the initialization formula."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestYaRNCacheExtension(CustomTestCase):
    def setUp(self):
        # Exercise cache construction without a serving context or accelerator.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        self.enterContext(
            patch(
                "sglang.srt.layers.rotary_embedding.base.publish_role",
                return_value=None,
            )
        )

    def make_rope(self, cls, dim=64, factor=1.0, dtype=torch.float32, **kwargs):
        if cls is DeepseekScalingRotaryEmbedding:
            kwargs["device"] = "cpu"
        return cls(dim, dim, 4096, 10000, True, factor, dtype, **kwargs)

    def expected_cache(self, rope, length):
        # Use the initialization frequency contract, not the extension helper.
        inv_freq = rope._compute_inv_freq(rope.scaling_factor)
        phase = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
        return torch.cat((phase.cos(), phase.sin()), dim=-1) * rope.mscale

    def test_extension(self):
        for cls in (YaRNScalingRotaryEmbedding, DeepseekScalingRotaryEmbedding):
            for dim in (64, 128):
                for factor in (1.0, 2.0):
                    with self.subTest(cls=cls.__name__, dim=dim, factor=factor):
                        rope = self.make_rope(cls, dim, factor)
                        original = rope.cos_sin_cache.clone()
                        initial_len = original.shape[0]
                        initial_cache = rope.cos_sin_cache
                        rope._ensure_cos_sin_cache_length(initial_len - 1)
                        self.assertIs(rope.cos_sin_cache, initial_cache)
                        for needed in (initial_len, initial_len + 4100):
                            prefix = rope.cos_sin_cache.clone()
                            rope._ensure_cos_sin_cache_length(needed)
                            self.assertGreater(rope.cos_sin_cache.shape[0], needed)
                            align = envs.SGLANG_ROPE_CACHE_ALIGN.get()
                            self.assertEqual(rope.cos_sin_cache.shape[0] % align, 0)
                            self.assertTrue(
                                torch.equal(prefix, rope.cos_sin_cache[: len(prefix)])
                            )
                            torch.testing.assert_close(
                                rope.cos_sin_cache,
                                self.expected_cache(rope, len(rope.cos_sin_cache)),
                                rtol=0,
                                atol=2e-7,
                            )
                        self.assertEqual(rope.max_position_embeddings, 4096)
                        if factor == 1:
                            inv = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
                            torch.testing.assert_close(
                                rope._compute_inv_freq(factor),
                                inv,
                                rtol=1e-6,
                                atol=1e-9,
                            )

    def test_nondefault_amplitude_and_cache_dtype(self):
        for cls in (YaRNScalingRotaryEmbedding, DeepseekScalingRotaryEmbedding):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(cls=cls.__name__, dtype=dtype):
                    rope = self.make_rope(
                        cls, factor=2.0, attn_factor=1.3, mscale=1.2, mscale_all_dim=0.4
                    )
                    # Match the runtime path that casts/moves an existing cache.
                    rope.cos_sin_cache = rope.cos_sin_cache.to(dtype)
                    prefix = rope.cos_sin_cache.clone()
                    rope._ensure_cos_sin_cache_length(8192)
                    self.assertEqual(rope.cos_sin_cache.dtype, dtype)
                    self.assertEqual(rope.cos_sin_cache.device.type, "cpu")
                    self.assertTrue(
                        torch.equal(prefix, rope.cos_sin_cache[: len(prefix)])
                    )
                    torch.testing.assert_close(
                        rope.cos_sin_cache,
                        self.expected_cache(rope, len(rope.cos_sin_cache)).to(dtype),
                        rtol=0,
                        atol=0,
                    )

    def test_deepseek_npu_auxiliary_tables(self):
        # Emulate table creation only; this does not exercise an NPU kernel.
        with patch("sglang.srt.layers.rotary_embedding.rope_variant._is_npu", True):
            rope = self.make_rope(
                DeepseekScalingRotaryEmbedding, factor=2.0, attn_factor=1.3
            )
        rope.cos_sin_cache = rope.cos_sin_cache.to(torch.bfloat16)
        for needed in (8192, 12288):
            old_cos, old_sin = (
                rope.cos_cached_total.clone(),
                rope.sin_cached_total.clone(),
            )
            rope._ensure_cos_sin_cache_length(needed)
            expected = self.expected_cache(rope, len(rope.cos_sin_cache))
            cos, sin = expected.chunk(2, dim=-1)
            for actual, want, old in (
                (rope.cos_cached_total, cos.repeat(1, 2), old_cos),
                (rope.sin_cached_total, sin.repeat(1, 2), old_sin),
            ):
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual[: len(old)], old))
                torch.testing.assert_close(actual, want, rtol=0, atol=2e-7)

    def test_default_rope(self):
        rope = RotaryEmbedding(64, 64, 4096, 10000, True, torch.float32)
        prefix = rope.cos_sin_cache.clone()
        rope._ensure_cos_sin_cache_length(8191)
        inv = 1 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
        phase = torch.outer(torch.arange(len(rope.cos_sin_cache)).float(), inv)
        self.assertTrue(torch.equal(prefix, rope.cos_sin_cache[:4096]))
        torch.testing.assert_close(
            rope.cos_sin_cache,
            torch.cat((phase.cos(), phase.sin()), dim=-1),
            rtol=0,
            atol=2e-7,
        )


if __name__ == "__main__":
    unittest.main()
