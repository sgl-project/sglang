"""CPU-only regression tests for CUDA IPC multimodal pool budgeting."""

import unittest

from sglang.srt.utils.cuda_ipc_transport_utils import (
    get_mm_feature_pool_size_per_worker,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCudaIpcPoolBudget(unittest.TestCase):
    def test_budget_is_not_multiplied_by_tokenizer_workers(self):
        budget = 1_024 * 1024 * 1024
        worker_num = 16

        per_worker = get_mm_feature_pool_size_per_worker(budget, worker_num)

        self.assertEqual(per_worker, 64 * 1024 * 1024)
        self.assertLessEqual(per_worker * worker_num, budget)

    def test_remainder_is_not_overallocated(self):
        self.assertEqual(get_mm_feature_pool_size_per_worker(1_001, 8), 125)
        self.assertLessEqual(get_mm_feature_pool_size_per_worker(1_001, 8) * 8, 1_001)

    def test_rejects_invalid_budget_or_worker_count(self):
        with self.assertRaisesRegex(ValueError, "total_pool_size"):
            get_mm_feature_pool_size_per_worker(0, 1)
        with self.assertRaisesRegex(ValueError, "tokenizer_worker_num"):
            get_mm_feature_pool_size_per_worker(1, 0)


if __name__ == "__main__":
    unittest.main()
