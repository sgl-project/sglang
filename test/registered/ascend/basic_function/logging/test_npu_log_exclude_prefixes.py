import unittest

import requests

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULogExcludePrefixes(TestNPULoggingBase):
    """Testcase: Verify the functionality of the --uvicorn-access-log-exclude-prefixes parameter.

    Description:
        Verifies that when the --uvicorn-access-log-exclude-prefixes parameter is configured with specified URL prefixes
        during server startup, HTTP requests matching these prefixes will NOT be recorded in the uvicorn access logs.
        Ensures the log filtering mechanism works correctly.

    [Test Category] Parameter
    [Test Target] --uvicorn-access-log-exclude-prefixes;
    """

    if_exclude_prefixes = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.if_exclude_prefixes:
            cls.other_args.extend(
                ["--uvicorn-access-log-exclude-prefixes", "/health", "/get_server_info"]
            )
        cls.launch_server()

    def test_log_exclude_prefixes(self):
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)
        response = requests.get(f"{self.base_url}/get_server_info", timeout=10)
        self.assertEqual(response.status_code, 200)
        self.out_log_file.seek(0)
        content = self.out_log_file.read()
        # The logs should not include health and server info request information when
        # --uvicorn-access-log-exclude-prefixes is set with health and server info prefix
        health_log = '"GET /health HTTP/1.1" 200 OK'
        server_info_log = '"GET /get_server_info HTTP/1.1" 200 OK'
        if self.if_exclude_prefixes:
            self.assertNotIn(health_log, content)
            self.assertNotIn(server_info_log, content)
        else:
            self.assertIn(health_log, content)
            self.assertIn(server_info_log, content)


class TestNPULogNotExcludePrefixes(TestNPULogExcludePrefixes, TestNPULoggingBase):
    if_exclude_prefixes = False


if __name__ == "__main__":
    unittest.main()
