import unittest

from sglang.srt.managers.cache_controller import (
    _should_skip_rank_replicated_backup,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestContextParallelBackupWriter(CustomTestCase):
    def test_every_context_parallel_shard_has_a_writer(self):
        self.assertFalse(_should_skip_rank_replicated_backup(True, 0))

    def test_attention_tp_replica_still_skips_backup(self):
        self.assertTrue(_should_skip_rank_replicated_backup(True, 1))

    def test_non_replicated_cache_never_skips_backup(self):
        self.assertFalse(_should_skip_rank_replicated_backup(False, 7))


if __name__ == "__main__":
    unittest.main()
