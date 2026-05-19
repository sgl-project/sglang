import json
import re
import unittest

from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULogRequestsFormatText(TestNPULoggingBase):
    """Testcase: Verify the functionality of --enable-request-time-stats-logging to generate Req Time Stats logs on Ascend backend with Llama-3.2-1B-Instruct model.

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    """

    log_requests_format = "text"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.other_args.extend(["--log-requests-format", cls.log_requests_format])
        cls.launch_server()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.output_capturer.stop()

    def test_format(self):
        self.inference_once()

        content = self.output_capturer.get_all()
        self.assertIn("Receive:", content, f"'Receive:' not found")
        self.assertIn("Finish:", content, f"'Finish:' not found")


class TestNPULogRequestsFormatJson(TestNPULogRequestsFormatText):
    log_requests_format = "json"

    def test_format(self):
        self.inference_once()

        content = self.output_capturer.get_all()
        ts_pattern = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}]\s*')
        received_found = False
        finished_found = False
        for line in content.splitlines():
            s = line.strip()
            if not s:
                continue

            # Match valid timestamps
            if s.startswith('['):
                s = ts_pattern.sub('', s)

            if not s.startswith('{'):
                continue
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                continue

            if data.get("event") == "request.received":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                received_found = True
            elif data.get("event") == "request.finished":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertIn("out", data)
                finished_found = True

        self.assertTrue(received_found, f"request.received event not found")
        self.assertTrue(finished_found, f"request.finished event not found")


if __name__ == "__main__":
    unittest.main()
