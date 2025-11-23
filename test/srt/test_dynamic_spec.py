"""
Unit tests for dynamic speculative decoding functionality.

This test verifies that the scheduler correctly switches between spec and non-spec
modes based on batch size thresholds by checking accept_length metrics changes.
"""

import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDynamicSpec(CustomTestCase):
    """Test dynamic speculative decoding with automatic mode switching."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_size_threshold = 8  # Threshold for testing
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                "2",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "4",
                "--enable-dynamic-spec",
                "--speculative-batch-size-threshold",
                str(cls.batch_size_threshold),
                "--max-running-requests",
                "16",
                "--mem-fraction-static",
                "0.7",
                "--decode-log-interval",
                "5",
                "--disable-cuda-graph",  # Disable cuda graph for dynamic spec
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mode_switching_with_metrics_verification(self):
        """Test mode switching by verifying spec_verify_ct changes."""
        resp = requests.get(self.base_url + "/flush_cache")
        self.assertEqual(resp.status_code, 200)

        # Step 1: Small batch to establish spec mode baseline
        url = self.base_url + "/generate"
        small_prompts = ["Tell me a short story"] * 4  # < threshold
        data_small = {
            "text": small_prompts,
            "sampling_params": {"temperature": 0, "max_new_tokens": 64},
        }
        response_small = requests.post(url, json=data_small)
        self.assertEqual(response_small.status_code, 200)
        outputs_small = response_small.json()

        # Verify spec mode by checking spec_verify_ct
        spec_count_small = sum(
            1
            for o in outputs_small
            if o.get("meta_info", {}).get("spec_verify_ct", 0) > 0
        )
        print(f"Small batch: {spec_count_small}/4 used spec (verify_ct > 0)")
        self.assertGreater(spec_count_small, 0, "Small batch should use spec mode")

        # Step 2: Get initial accept length
        server_info_before = requests.get(self.base_url + "/get_server_info").json()
        accept_len_before = server_info_before["internal_states"][0].get(
            "avg_spec_accept_length", 0
        )

        # Step 3: Large batch to trigger switch to non-spec
        large_prompts = ["Hello"] * 12  # > threshold
        data_large = {
            "text": large_prompts,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        }
        response_large = requests.post(url, json=data_large)
        self.assertEqual(response_large.status_code, 200)
        outputs_large = response_large.json()

        # In non-spec mode, spec_verify_ct should be 0 or not exist
        spec_count_large = sum(
            1
            for o in outputs_large
            if o.get("meta_info", {}).get("spec_verify_ct", 0) > 0
        )
        print(f"Large batch: {spec_count_large}/12 used spec (verify_ct > 0)")
        # Large batch should use non-spec, so spec_count should be 0
        self.assertEqual(spec_count_large, 0, "Large batch should use non-spec mode")

        # Step 4: Small batch again to verify switch back to spec
        time.sleep(0.5)  # Brief pause
        response_small2 = requests.post(url, json=data_small)
        self.assertEqual(response_small2.status_code, 200)
        outputs_small2 = response_small2.json()

        spec_count_small2 = sum(
            1
            for o in outputs_small2
            if o.get("meta_info", {}).get("spec_verify_ct", 0) > 0
        )
        print(f"Small batch again: {spec_count_small2}/4 used spec (verify_ct > 0)")
        self.assertGreater(spec_count_small2, 0, "Should switch back to spec mode")


if __name__ == "__main__":
    unittest.main()
