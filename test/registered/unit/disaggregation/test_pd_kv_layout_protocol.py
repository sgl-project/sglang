"""CPU tests for the pool-declared PD KV layout (``KVCache.pd_kv_layout``)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


import unittest

from sglang.srt.disaggregation.utils import get_pd_kv_layout
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    KVCache,
    MHATokenToKVPool,
)
from sglang.test.test_utils import CustomTestCase


class _OutOfTreePool(MHATokenToKVPool):
    """Stand-in for a platform pool with a different byte layout."""

    def __init__(self):  # skip buffer allocation
        pass

    def pd_kv_layout(self):
        return "head_major_in_page"


class TestPdKvLayoutProtocol(CustomTestCase):
    def test_default_pool_declares_nothing(self):
        pool = object.__new__(MHATokenToKVPool)
        self.assertIsNone(pool.pd_kv_layout())
        self.assertIsNone(get_pd_kv_layout(pool))

    def test_subclass_declares_its_layout(self):
        self.assertEqual(get_pd_kv_layout(_OutOfTreePool()), "head_major_in_page")

    def test_hybrid_linear_pool_forwards_the_full_attention_pool(self):
        hybrid = object.__new__(HybridLinearKVPool)
        hybrid.full_kv_pool = _OutOfTreePool()
        self.assertEqual(get_pd_kv_layout(hybrid), "head_major_in_page")
        hybrid.full_kv_pool = object.__new__(MHATokenToKVPool)
        self.assertIsNone(get_pd_kv_layout(hybrid))

    def test_objects_without_the_method_declare_nothing(self):
        self.assertIsNone(get_pd_kv_layout(object()))
        self.assertIsNone(get_pd_kv_layout(None))

    def test_deepseek_v4_pool_still_declares_through_the_protocol(self):
        self.assertTrue(issubclass(DeepSeekV4TokenToKVPool, KVCache))
        self.assertIsNot(DeepSeekV4TokenToKVPool.pd_kv_layout, KVCache.pd_kv_layout)


if __name__ == "__main__":
    unittest.main()
