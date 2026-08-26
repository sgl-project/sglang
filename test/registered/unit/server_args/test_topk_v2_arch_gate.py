import unittest
from unittest.mock import patch

from sglang.srt import server_args as server_args_module
from sglang.srt.environ import envs
from sglang.srt.server_args import _disable_topk_v2_without_thread_block_clusters
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTopkV2ArchGate(unittest.TestCase):
    def setUp(self):
        self._previous = envs.SGLANG_OPT_USE_TOPK_V2.get()
        envs.SGLANG_OPT_USE_TOPK_V2.set(True)

    def tearDown(self):
        envs.SGLANG_OPT_USE_TOPK_V2.set(self._previous)

    def test_disabled_below_sm90(self):
        with patch.object(server_args_module, "is_sm80_supported", return_value=True):
            _disable_topk_v2_without_thread_block_clusters()
        self.assertFalse(envs.SGLANG_OPT_USE_TOPK_V2.get())

    def test_left_alone_from_sm90_on(self):
        with patch.object(server_args_module, "is_sm80_supported", return_value=False):
            _disable_topk_v2_without_thread_block_clusters()
        self.assertTrue(envs.SGLANG_OPT_USE_TOPK_V2.get())

    def test_logs_the_reason(self):
        with patch.object(server_args_module, "is_sm80_supported", return_value=True):
            with self.assertLogs("sglang.srt.server_args", level="INFO") as logs:
                _disable_topk_v2_without_thread_block_clusters()
        self.assertTrue(
            any("thread-block clusters" in line for line in logs.output), logs.output
        )


if __name__ == "__main__":
    unittest.main()
