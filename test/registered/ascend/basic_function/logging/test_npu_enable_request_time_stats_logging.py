import unittest

from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPUEnableRequestTimeStatsLogging(TestNPULoggingBase):
    """Testcase: Verify the functionality of --enable-request-time-stats-logging to generate Req Time Stats logs.

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.other_args.extend(["--enable-request-time-stats-logging"])
        cls.launch_server()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.output_capturer.stop()

    def test_enable_request_time_stats_logging(self):
        self.inference_once()

        self.assertIn(
            "Req Time Stats",
            self.output_capturer.get_all(),
            f"Keyword not found in server logs.",
        )


if __name__ == "__main__":
    unittest.main()
