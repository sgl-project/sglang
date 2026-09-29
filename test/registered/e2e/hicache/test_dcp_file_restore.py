import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.hicache_fixture import FileHiCacheRestoreMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


class TestDcpFileRestore(FileHiCacheRestoreMixin, CustomTestCase):
    pass


if __name__ == "__main__":
    unittest.main()
