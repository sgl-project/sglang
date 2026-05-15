import math
import unittest

from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestDecodeLogInterval(TestNPULoggingBase):
    """Testcase: Verify that configuration --decode-log-interval can correctly record decode information in batches.

    [Test Category] Parameter
    [Test Target] --decode-log-interval
    """

    decode_numbers = 10

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.other_args.extend(["--decode-log-interval", cls.decode_numbers])
        cls.launch_server()

    def test_decode_log_interval(self):
        max_tokens = 512
        self.inference_once(max_tokens=max_tokens)
        self.out_log_file.seek(0)
        self.err_log_file.seek(0)
        content = self.out_log_file.read() + self.err_log_file.read()
        decod_batch_count = content.count("Decode batch")
        expected_decod_batch_count = math.floor((max_tokens + 9) / self.decode_numbers)
        self.assertEqual(decod_batch_count, expected_decod_batch_count)


class TestDecodeLogIntervalOther(TestDecodeLogInterval):
    decode_numbers = 30


if __name__ == "__main__":
    unittest.main()
