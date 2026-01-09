import asyncio
import os
import unittest
from typing import Any, List, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_with_custom_params,
)


class TestRoutingKeyScheduling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",
                "2",
                "--schedule-policy",
                "routing-key",
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

    def test_routing_key_scheduling_order(self):
        """Verify requests with matching routing keys are prioritized."""
        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "routing_key": "key_a",
                        "sampling_params": {"max_new_tokens": 3000},
                    },
                    {
                        "routing_key": "key_b",
                        "sampling_params": {"max_new_tokens": 10},
                    },
                    {
                        "routing_key": "key_a",
                        "sampling_params": {"max_new_tokens": 10},
                    },
                ],
            )
        )

        e2e_latencies = []
        for got_status, got_json in responses:
            self.assertEqual(got_status, 200)
            e2e_latencies.append(got_json["meta_info"]["e2e_latency"])

        self.assertLess(
            e2e_latencies[2],
            e2e_latencies[1],
            "Request with matching routing_key (key_a) should finish before key_b",
        )


if __name__ == "__main__":
    unittest.main()
