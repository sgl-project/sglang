import asyncio
import os
import re
import unittest
from typing import Any, List, Optional, Tuple

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


class TestPriorityScheduling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",  # Enforce max request concurrency is 1
                "1",
                "--max-queued-requests",  # Enforce max queued request number is 3
                "3",
                "--enable-priority-scheduling",  # Enable priority scheduling
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        _verify_max_running_requests_and_max_queued_request_validation(1, 3)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_priority_scheduling_request_ordering_validation(self):
        """Verify pending requests are ordered by priority and received timestamp."""

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "priority": 0,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # starts being processed first
                    {"priority": 1},  # third
                    {"priority": 1},  # fourth
                    {"priority": 2},  # second
                ],
            )
        )

        expected_status_and_error_messages = [
            (200, None),
            (200, None),
            (200, None),
            (200, None),
        ]

        e2e_latencies = []
        _verify_genereate_responses(
            responses, expected_status_and_error_messages, e2e_latencies
        )
        assert e2e_latencies[0] < e2e_latencies[3] < e2e_latencies[1] < e2e_latencies[2]

    def test_priority_scheduling_existing_requests_abortion_validation(self):
        """Verify lower priority requests are aborted when incoming requests have higher priority"""

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "priority": 1,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # starts being processed first and holds the running queue capacity
                    {"priority": 2},  # aborted by request 5
                    {"priority": 3},  # aborted by request 6
                    {"priority": 4},  # aborted by request 7
                    {"priority": 5},  # fourth
                    {"priority": 6},  # third
                    {"priority": 7},  # second
                ],
            )
        )

        expected_status_and_error_messages = [
            (200, None),
            (503, "The request is aborted by a higher priority request."),
            (503, "The request is aborted by a higher priority request."),
            (503, "The request is aborted by a higher priority request."),
            (200, None),
            (200, None),
            (200, None),
        ]

        e2e_latencies = []
        _verify_genereate_responses(
            responses, expected_status_and_error_messages, e2e_latencies
        )
        assert e2e_latencies[0] < e2e_latencies[6] < e2e_latencies[5] < e2e_latencies[4]

    def test_priority_scheduling_incoming_request_rejection_validation(self):
        """Verify incoming requests are rejected when existing requests have higher priority"""

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "priority": 7,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # starts being processed first and holds the running queue capacity
                    {"priority": 6},  # second
                    {"priority": 5},  # third
                    {"priority": 4},  # fourth
                    {"priority": 3},  # rejected
                    {"priority": 2},  # rejected
                    {"priority": 1},  # rejected
                ],
            )
        )

        expected_status_and_error_messages = [
            (200, None),
            (200, None),
            (200, None),
            (200, None),
            (503, "The request queue is full."),
            (503, "The request queue is full."),
            (503, "The request queue is full."),
        ]

        e2e_latencies = []
        _verify_genereate_responses(
            responses, expected_status_and_error_messages, e2e_latencies
        )
        assert e2e_latencies[0] < e2e_latencies[1] < e2e_latencies[2] < e2e_latencies[3]

    def test_priority_scheduling_preemption_meeting_threshold_validation(self):
        """Verify running requests are preempted by requests with priorities meeting the preemption threshold"""

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "priority": 0,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # starts being processed first then preempted or pushed by later requests, and finishes last.
                    {
                        "priority": 10,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # scheduled after the third request, and finishes second.
                    {
                        "priority": 20,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # finishes first.
                ],
            )
        )

        expected_status_and_error_messages = [
            (200, None),
            (200, None),
            (200, None),
        ]

        e2e_latencies = []
        _verify_genereate_responses(
            responses, expected_status_and_error_messages, e2e_latencies
        )

        assert e2e_latencies[2] < e2e_latencies[1] < e2e_latencies[0]

    def test_priority_scheduling_preemption_below_threshold_validation(self):
        """Verify running requests are not preempted by requests with priorities below preemption threshold"""

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "priority": 0,
                        "sampling_params": {"max_new_tokens": 10000},
                    },
                    {
                        "priority": 5,
                        "sampling_params": {"max_new_tokens": 10000},
                    },
                ],
            )
        )

        expected_status_and_error_messages = [
            (200, None),
            (200, None),
        ]

        e2e_latencies = []
        _verify_genereate_responses(
            responses, expected_status_and_error_messages, e2e_latencies
        )

        assert e2e_latencies[0] < e2e_latencies[1]


class TestPrioritySchedulingMultipleRunningRequests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",  # Enforce max request concurrency is 2
                "2",
                "--max-queued-requests",  # Enforce max queued request number is 3
                "3",
                "--enable-priority-scheduling",  # Enable priority scheduling
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        _verify_max_running_requests_and_max_queued_request_validation(2, 3)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_priority_scheduling_with_multiple_running_requests_preemption(self):
        """Verify preempting a subset of running requests is safe."""

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "priority": 10,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # finishes first
                    {
                        "priority": 5,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # preempted by fourth request, then finishes third
                    {
                        "priority": 15,
                        "sampling_params": {"max_new_tokens": 10000},
                    },  # preempt the first request
                ],
            )
        )

        expected_status_and_error_messages = [
            (200, None),
            (200, None),
            (200, None),
            (200, None),
        ]

        _verify_genereate_responses(responses, expected_status_and_error_messages, [])


def _verify_genereate_responses(
    responses: Tuple[int, Any, float],
    expected_code_and_error_message: Tuple[int, Any],
    e2e_latencies: List[Optional[float]],
):
    """
    Verify generate response results are as expected based on status code and response json object content.
    In addition, collects e2e latency info to verify scheduling and processing ordering.
    """
    for got, expected in zip(responses, expected_code_and_error_message):
        got_status, got_json = got
        expected_status, expected_err_msg = expected

        # Check status code is as expected
        assert got_status == expected_status

        # Check error message content or fields' existence based on status code
        if got_status != 200:
            assert got_json["object"] == "error"
            assert got_json["message"] == expected_err_msg
        else:
            assert "object" not in got_json
            assert "message" not in got_json

        # Collect e2e latencies for scheduling validation
        e2e_latencies.append(
            got_json["meta_info"]["e2e_latency"] if got_status == 200 else None
        )


def _verify_max_running_requests_and_max_queued_request_validation(
    max_running_requests: int, max_queued_requests: int
):
    """Verify running request and queued request numbers based on server logs."""
    rr_pattern = re.compile(r"#running-req:\s*(\d+)")
    qr_pattern = re.compile(r"#queue-req:\s*(\d+)")

    with open(STDERR_FILENAME) as lines:
        for line in lines:
            rr_match, qr_match = rr_pattern.search(line), qr_pattern.search(line)
            if rr_match:
                assert int(rr_match.group(1)) <= max_running_requests
            if qr_match:
                assert int(qr_match.group(1)) <= max_queued_requests


if __name__ == "__main__":
    unittest.main()
