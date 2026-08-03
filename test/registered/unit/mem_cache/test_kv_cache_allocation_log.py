"""Unit tests for KV cache allocation log labels."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import GB, KVCache, MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKVCacheAllocationLog(CustomTestCase):
    def test_allocation_label_and_generic_fallback(self):
        pool = object.__new__(MHATokenToKVPool)
        pool.dtype = torch.bfloat16
        pool.get_kv_size_bytes = lambda: (2 * GB, 3 * GB)

        for label, expected_name in (
            (None, "KV Cache"),
            ("Full", "Full KV Cache"),
            ("SWA", "SWA KV Cache"),
        ):
            with self.subTest(label=label):
                pool.allocation_label = label
                with self.assertLogs(
                    "sglang.srt.mem_cache.memory_pool", level="INFO"
                ) as logs:
                    KVCache._finalize_allocation_log(pool, 123)

                self.assertIn(
                    f"{expected_name} is allocated. dtype: torch.bfloat16, "
                    "#tokens: 123, K size: 2.00 GB, V size: 3.00 GB",
                    logs.output[0],
                )

    @patch(
        "sglang.srt.mem_cache.swa_memory_pool.maybe_init_custom_mem_pool",
        return_value=(False, None, None),
    )
    def test_swa_pool_labels_full_then_swa(self, _mock_custom_pool):
        allocations = []

        class FakeKVPool:
            def __init__(self, *, size, allocation_label, **_kwargs):
                allocations.append((allocation_label, size))

            def get_kv_size_bytes(self):
                return 0, 0

        SWAKVPool(
            size=300,
            size_swa=200,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[0, 1],
            full_attention_layer_ids=[2],
            device="cpu",
            token_to_kv_pool_class=FakeKVPool,
        )

        self.assertEqual(allocations, [("Full", 300), ("SWA", 200)])


if __name__ == "__main__":
    unittest.main()
