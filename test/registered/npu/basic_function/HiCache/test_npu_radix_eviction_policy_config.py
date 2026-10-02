import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNPURadixEvictionPolicyConfig(CustomTestCase):
    """Testcase: --radix-eviction-policy-config smoke test on Ascend NPU.

    Covers that the slru eviction policy accepts its tuning parameter
    --radix-eviction-policy-config '{"protected_threshold": 4}' and that
    the radix cache (prefix cache) keeps working under this configuration.

    [Test Category] HiCache
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            1,
            "--radix-eviction-policy",
            "slru",
            "--radix-eviction-policy-config",
            '{"protected_threshold": 4}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def tearDown(self):
        try:
            # Call the '/flush_cache' interface to clear RadixCache.
            response = requests.post(f"{DEFAULT_URL_FOR_TEST}/flush_cache")
            self.assertEqual(response.status_code, 200, "Failed to flush cache")
        except Exception as e:
            self.fail(f"Flush cache failed with error: {str(e)}")

    def test_radix_eviction_policy_config_accepted(self):
        """Server starts and serves requests with the config enabled"""
        # Long identical prompt (meets page size requirement)
        long_text = "What is the capital of France?" * 36
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("meta_info", response.json())

    def test_radix_cache_reused_long_identical_under_slru(self):
        """RadixCache is still reused under slru + protected_threshold config"""
        # Ultra-long repeated prompt (meets page size requirement)
        long_text = "What is the capital of France?" * 36
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": long_text,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 10,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            if i == 0:
                # First request: no cache
                self.assertEqual(cached_tokens, 0)
            else:
                # Second request: cache reused, which drives node hit_count
                # feeding the slru protection logic
                self.assertGreater(cached_tokens, 0)


if __name__ == "__main__":
    unittest.main()