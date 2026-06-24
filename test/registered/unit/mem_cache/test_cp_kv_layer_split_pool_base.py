"""Unit tests for CP KV LayerSplit pool helpers."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.cp_kv_layer_split.pool_base import CpKvLayerSplitPoolBase
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCpKvLayerSplitPoolBase(CustomTestCase):
    def test_pool_num_pages_includes_dummy_page_namespace(self):
        self.assertEqual(
            CpKvLayerSplitPoolBase._pool_num_pages(
                SimpleNamespace(size=256, page_size=128)
            ),
            3,
        )

    def test_pool_num_pages_uses_pool_page_namespace(self):
        self.assertEqual(
            CpKvLayerSplitPoolBase._pool_num_pages(
                SimpleNamespace(size=512, page_size=128)
            ),
            5,
        )
        self.assertEqual(
            CpKvLayerSplitPoolBase._pool_num_pages(
                SimpleNamespace(size=257, page_size=128)
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
