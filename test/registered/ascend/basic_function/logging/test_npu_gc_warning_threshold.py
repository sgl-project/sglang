import threading
import unittest

import requests

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPUGCWarningThreshold(TestNPULoggingBase):
    """Testcase: Verify the functionality of the --gc-warning-threshold-secs parameter

    [Description]
        Verifies the alert threshold setting function of this parameter: this parameter configures the maximum allowed
        duration threshold for Garbage Collection (GC) operations. When the actual execution duration of a GC operation
        **exceeds** this threshold, the system must proactively log an alert-level message to indicate abnormal GC performance;
        if the GC duration does not exceed the threshold, no such alert log should be generated, ensuring precise alert
        rule triggering.

    [Test Category] Parameter
    [Test Target] --gc-warning-threshold-secs
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # overwrite other_args to reduce log printing by removing --log-requests
        cls.other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        # --gc-warning-threshold-secs=0.01 (a tiny value) is set to ensure GC duration exceeds the alarm threshold.
        cls.other_args.extend(["--gc-warning-threshold-secs", "0.01"])
        cls.launch_server()

    def test_gc_warning_threshold(self):
        """Validate SGLang logs GC warnings when GC duration exceeds --gc-warning-threshold-secs threshold.

        Core Functionality:
            1. Generate high-concurrency requests with long sequences to create large temporary objects in SGLang service
            2. Trigger garbage collection (GC) by overwhelming the service with memory-intensive requests
            3. Verify that when GC time exceeds the configured threshold, a specific GC warning log is recorded in the error log file
        """
        prompt_template = (
            "just return me a string with of 10000 characters: " + "A" * 10000
        )
        max_token = 10000

        def send_request():
            requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt_template,
                    "sampling_params": {"temperature": 0, "max_new_tokens": max_token},
                },
            )

        threads = []
        for _ in range(2000):
            t = threading.Thread(target=send_request)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        GC_info = "LONG GARBAGE COLLECTION DETECTED"
        self.out_log_file.seek(0)
        self.err_log_file.seek(0)
        content = self.out_log_file.read() + self.err_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIn(GC_info, content)


if __name__ == "__main__":
    unittest.main()
