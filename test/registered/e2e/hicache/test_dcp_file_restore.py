import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.hicache_fixture import FileHiCacheRestoreMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


class TestDcpFileRestore(FileHiCacheRestoreMixin, CustomTestCase):
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    revision = "85864749cd611b4353ce1decdb286193298f64c7"
    tp = 2
    dcp = 2
    hybrid = False
    other_args = ["--attention-backend", "flashinfer", "--dcp-comm-backend", "ag_rs"]


if __name__ == "__main__":
    unittest.main()
