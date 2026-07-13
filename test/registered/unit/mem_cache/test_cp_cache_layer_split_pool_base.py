"""Unit tests for CP Cache LayerSplit pool helpers."""

import unittest

from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
    CpCacheLayerSplitPoolBase,
    is_cp_cache_layer_split_pool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Pool(CpCacheLayerSplitPoolBase):
    def __init__(self, cp_rank=0, cp_size=2, start_layer=0, layer_num=4):
        super().__init__(
            cp_rank=cp_rank,
            cp_size=cp_size,
            layer_shard_start_layer=start_layer,
            layer_shard_layer_num=layer_num,
        )


class TestCpCacheLayerSplitPoolBase(CustomTestCase):
    def test_pool_is_marked_as_layer_split_without_descriptor_requirement(self):
        pool = _Pool()

        self.assertTrue(is_cp_cache_layer_split_pool(pool))
        self.assertFalse(pool.requires_descriptor_matched_transfer)

    def test_invalid_cp_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid cp_rank"):
            _Pool(cp_rank=2, cp_size=2)


if __name__ == "__main__":
    unittest.main()
