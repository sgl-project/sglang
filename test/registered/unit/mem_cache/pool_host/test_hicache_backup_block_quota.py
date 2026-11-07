"""SGLANG_HICACHE_BACKUP_BLOCK_QUOTA only adds ``block_quota`` to the backup copy when set."""

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import hicache_backup_copy_kwargs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiCacheBackupBlockQuota(CustomTestCase):
    def test_unset_leaves_the_kernel_default(self):
        with envs.SGLANG_HICACHE_BACKUP_BLOCK_QUOTA.override(None):
            self.assertEqual(hicache_backup_copy_kwargs(), {})

    def test_set_passes_block_quota(self):
        with envs.SGLANG_HICACHE_BACKUP_BLOCK_QUOTA.override(16):
            self.assertEqual(hicache_backup_copy_kwargs(), {"block_quota": 16})


if __name__ == "__main__":
    unittest.main()
