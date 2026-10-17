import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerSparseIndexAlignment(CustomTestCase):
    @staticmethod
    def _compressed_qsa_config(ratio: int):
        return SimpleNamespace(
            model_type="qwen4_exp",
            architectures=["Qwen4ExpForCausalLM"],
            indexer_n_heads=16,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=ratio,
        )

    def test_compressed_qsa_sets_truncation_alignment(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.truncation_align_size = None
        scheduler.model_config = SimpleNamespace(
            hf_config=self._compressed_qsa_config(4)
        )

        scheduler.init_sparse_index_truncation_align()

        self.assertEqual(scheduler.truncation_align_size, 4)

    def test_qsa_alignment_preserves_existing_alignment_with_lcm(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.truncation_align_size = 6
        scheduler.model_config = SimpleNamespace(
            hf_config=self._compressed_qsa_config(4)
        )

        scheduler.init_sparse_index_truncation_align()

        self.assertEqual(scheduler.truncation_align_size, 12)


if __name__ == "__main__":
    unittest.main()
