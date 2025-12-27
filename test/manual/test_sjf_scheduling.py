"""
Test SJF (Shortest Job First) scheduling policy.

This policy is designed for prefill-only nodes in disaggregated serving (PD disaggregation),
where prioritizing shorter requests can minimize the average/mean TTFT by reducing
overall queue waiting time.
"""

import asyncio
import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_with_custom_params,
)


class TestSJFScheduling(CustomTestCase):
    """Test SJF (Shortest Job First) scheduling policy."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",
                "1",  # Enforce sequential processing to observe ordering
                "--schedule-policy",
                "sjf",  # Use SJF scheduling
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_sjf_scheduling_shorter_requests_first(self):
        """Verify shorter input requests are processed before longer ones."""

        # Create requests with different input lengths
        # The first request blocks while others queue up
        short_prompt = "Hi"
        medium_prompt = "Hello, how are you doing today? I hope you are well."
        long_prompt = "Please write a very detailed and comprehensive essay about the history of artificial intelligence, covering all major milestones, key researchers, and technological breakthroughs from the 1950s to the present day."

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    # First request: blocks the queue
                    {
                        "prompt": long_prompt,
                        "sampling_params": {"max_new_tokens": 50},
                    },
                    # These requests arrive while first is processing
                    # With SJF, shorter ones should complete first
                    {
                        "prompt": long_prompt,  # longest queued
                        "sampling_params": {"max_new_tokens": 5},
                    },
                    {
                        "prompt": short_prompt,  # shortest queued
                        "sampling_params": {"max_new_tokens": 5},
                    },
                    {
                        "prompt": medium_prompt,  # medium queued
                        "sampling_params": {"max_new_tokens": 5},
                    },
                ],
            )
        )

        # All requests should succeed
        for resp in responses:
            self.assertEqual(resp["status_code"], 200, f"Request failed: {resp}")

        # Extract e2e latencies (end-to-end time for each request)
        e2e_latencies = [resp["e2e_latency"] for resp in responses]

        # The first request (index 0) finishes processing first (blocking request)
        # After that, with SJF scheduling:
        # - short_prompt (index 2) should finish before medium_prompt (index 3)
        # - medium_prompt (index 3) should finish before long_prompt (index 1)

        # Since they queue while request 0 is running, we check the order of 1,2,3
        # Request 2 (short) < Request 3 (medium) < Request 1 (long)
        self.assertLess(
            e2e_latencies[2],
            e2e_latencies[3],
            f"Short request should complete before medium request. "
            f"Short latency: {e2e_latencies[2]}, Medium latency: {e2e_latencies[3]}",
        )
        self.assertLess(
            e2e_latencies[3],
            e2e_latencies[1],
            f"Medium request should complete before long request. "
            f"Medium latency: {e2e_latencies[3]}, Long latency: {e2e_latencies[1]}",
        )


class TestSJFWithPriorityScheduling(CustomTestCase):
    """Test SJF scheduling combined with priority scheduling."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",
                "1",
                "--schedule-policy",
                "sjf",
                "--enable-priority-scheduling",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_sjf_with_priority_scheduling(self):
        """Verify priority takes precedence, then SJF within same priority."""

        short_prompt = "Hi"
        long_prompt = "Please write a very detailed and comprehensive essay about the history of artificial intelligence."

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    # First request: blocks the queue
                    {
                        "prompt": long_prompt,
                        "sampling_params": {"max_new_tokens": 50},
                        "priority": 0,
                    },
                    # Lower priority (processed later), long
                    {
                        "prompt": long_prompt,
                        "sampling_params": {"max_new_tokens": 5},
                        "priority": 1,
                    },
                    # Higher priority (processed first among queued), short
                    {
                        "prompt": short_prompt,
                        "sampling_params": {"max_new_tokens": 5},
                        "priority": 2,
                    },
                    # Lower priority, short (should be after priority 2 but before long with same priority)
                    {
                        "prompt": short_prompt,
                        "sampling_params": {"max_new_tokens": 5},
                        "priority": 1,
                    },
                ],
            )
        )

        # All requests should succeed
        for resp in responses:
            self.assertEqual(resp["status_code"], 200, f"Request failed: {resp}")

        e2e_latencies = [resp["e2e_latency"] for resp in responses]

        # Request 0 finishes first (blocking)
        # Then Request 2 (highest priority among queued)
        # Then within priority 1: Request 3 (short) before Request 1 (long)

        self.assertLess(
            e2e_latencies[2],
            e2e_latencies[3],
            f"Higher priority request should complete first. "
            f"Priority 2 latency: {e2e_latencies[2]}, Priority 1 latency: {e2e_latencies[3]}",
        )
        self.assertLess(
            e2e_latencies[3],
            e2e_latencies[1],
            f"Within same priority, shorter request should complete first. "
            f"Short latency: {e2e_latencies[3]}, Long latency: {e2e_latencies[1]}",
        )


if __name__ == "__main__":
    unittest.main()
