import unittest
from unittest import mock

import torch

from sglang.srt.layers.rotary_embedding import base
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHipFallbackCacheDtype(CustomTestCase):
    """The HIP rotary fallback must get the cache in the query dtype.

    HIP keeps cos_sin_cache fp32 for the fused QSA indexer, but the fallback
    kernel reads it as the query dtype. Passing the fp32 buffer fails CUDA graph
    capture with "expected scalar type BFloat16 but found Float".
    """

    def _rope(self, dtype):
        patches = (
            mock.patch.object(base, "_is_cpu", True),
            mock.patch.object(base, "_is_hip", True),
            mock.patch.object(base, "publish_role", lambda: None),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        rope = base.RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=256,
            base=10000,
            is_neox_style=True,
            dtype=dtype,
        )
        rope.use_fallback_kernel = True
        rope.fallback_rotary_embedding = mock.Mock()
        return rope

    def test_fallback_gets_query_dtype_cache(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                rope = self._rope(dtype)
                fp32_cache = rope.cos_sin_cache
                self.assertEqual(fp32_cache.dtype, torch.float32)

                positions = torch.arange(4)
                query = torch.zeros(4, 2 * 64, dtype=dtype)
                key = torch.zeros(4, 64, dtype=dtype)
                for _ in range(2):
                    rope.forward_cuda(positions, query, key)

                first, second = (
                    call.args[4]
                    for call in rope.fallback_rotary_embedding.call_args_list
                )
                self.assertEqual(first.dtype, dtype)
                self.assertTrue(torch.equal(first, fp32_cache.to(dtype)))
                # The cast copy is reused, and the fp32 buffer is left alone.
                self.assertIs(second, first)
                self.assertIs(rope.cos_sin_cache, fp32_cache)

                # Extending the buffer invalidates the cast copy.
                rope._ensure_cos_sin_cache_length(len(fp32_cache) + 16)
                rope.forward_cuda(positions, query, key)
                third = rope.fallback_rotary_embedding.call_args.args[4]
                self.assertEqual(rope.cos_sin_cache.dtype, torch.float32)
                self.assertEqual(third.dtype, dtype)
                self.assertTrue(torch.equal(third, rope.cos_sin_cache.to(dtype)))


if __name__ == "__main__":
    unittest.main()
