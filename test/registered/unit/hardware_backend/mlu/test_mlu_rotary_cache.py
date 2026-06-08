import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMLURotaryCacheTransform(CustomTestCase):
    def test_transform_cache_interleaved_layout(self):
        from sglang.srt.layers.rotary_embedding.base import _transform_cache

        cache = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        cos, sin = _transform_cache(cache, is_neox_style=False)

        self.assertEqual(cos.tolist(), [[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])
        self.assertEqual(sin.tolist(), [[4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0]])

    def test_transform_cache_neox_layout(self):
        from sglang.srt.layers.rotary_embedding.base import _transform_cache

        cache = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        cos, sin = _transform_cache(cache, is_neox_style=True)

        self.assertEqual(cos.tolist(), [[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]])
        self.assertEqual(sin.tolist(), [[4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0, 7.0]])


if __name__ == "__main__":
    unittest.main()
