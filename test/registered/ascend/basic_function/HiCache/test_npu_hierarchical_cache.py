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

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHierarchicalCache(CustomTestCase):
    """Testcase: HierarchicalCache Test on Ascend NPU.
    Cover scenarios:
    1. Long identical texts: cache can be reused
    2. Short identical texts: cache cannot be reused (page size limit)
    3. Different long texts: cache cannot be reused (prefix mismatch)

    [Test Category] HiCache
    [Test Target] --enable-hierarchical-cache
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.prefill_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            1,
            "--enable-hierarchical-cache",
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

    def test_hierarchical_cache_reused_long_identical(self):
        """Long identical texts should reuse HierarchicalCache"""
        # Ultra-long repeated prompt (meets page size requirement)
        long_text = "What is The capital of France?" * 36
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
                # Second request: cache reused
                self.assertGreater(cached_tokens, 0)

    def test_hierarchical_cache_not_reused_short_identical(self):
        """Short identical texts should NOT reuse HierarchicalCache (page size limit)"""
        # Short text prompt (does not meet page size requirement)
        short_text = "who am i?"
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": short_text,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 10,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            # No cache reuse for both requests
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            self.assertEqual(cached_tokens, 0)

    def test_hierarchical_cache_not_reused_different_long(self):
        """Different long texts should NOT reuse HierarchicalCache (text uniqueness)"""
        # Two different long text prompts (both meet the page size requirement)
        texts = [
            "Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?"
            * 8,
            "Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of shorts costs $16.50. One pair of pants costs $22.50 and one pair of shoes costs $42. How many dollars did Mishka spend on all the clothing items?"
            * 8,
        ]
        for text in texts:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": text,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 10,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            # No cache reuse for different text requests
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            self.assertEqual(cached_tokens, 0)


if __name__ == "__main__":
    unittest.main()
