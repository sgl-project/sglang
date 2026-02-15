import asyncio
import os
import re
import unittest
from typing import Any, List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_with_custom_params,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLowPriorityFirstScheduling(CustomTestCase):
    """Testcase: Verify the low-priority-first scheduling mechanism and its abortion logic by sending requests with different priorities and observing whether the behavior meets expectations, .

    [Test Category] Parameter
    [Test Target] --enable-priority-scheduling;--schedule-low-priority-values-first;--max-running-requests;--max-queued-requests
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-running-requests",
                "1",  # Limit concurrent running requests to 1 (enforce queueing)
                "--max-queued-requests",
                "3",  # Limit maximum queued requests to 3 (enforce request abortion when queue is full)
                "--enable-priority-scheduling",  # Enable priority-based request scheduling (prerequisite for priority features)
                "--schedule-low-priority-values-first",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        _verify_max_running_requests_and_max_queued_request_validation(1, 3)
        cls.stdout.close()
        cls.stderr.close()
        if os.path.exists(STDOUT_FILENAME):
            os.remove(STDOUT_FILENAME)
        if os.path.exists(STDERR_FILENAME):
            os.remove(STDERR_FILENAME)

    def test_low_priority_value_first_ordering(self):
        """Test core scheduling logic: lower priority values execute first

        Test Scenario:
        - Send 4 concurrent requests with priorities [0, 4, 2, 1]
        - Expected execution order: 0 (first) → 1 → 2 → 4 (last)
        - Validate via e2e latencies (earlier execution = lower latency value)
        """

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {"priority": 0, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 4, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 2, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 1, "sampling_params": {"max_new_tokens": 100}},
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

        assert e2e_latencies[0] < e2e_latencies[3] < e2e_latencies[2] < e2e_latencies[1]

    def test_low_priority_first_abortion_logic(self):
        """Test abortion logic: high-priority values abort lower-priority queued requests (reverse of execution order)

        Test Scenario:
        - Send 7 concurrent requests with priorities [10, 9, 8, 7, 6, 5, 0]
        - --max-running-requests=1, --max-queued-requests=3: Only 1 running + 3 queued = 4 requests can be processed
        - High priority values (for abortion: 10 > 9 > 8 > ...) → lower priority values in queue are aborted
        - Expected: Priority 10 (running), 0/5/6 (queued, lower values = higher execution priority), 7/8/9 (aborted)
        """

        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {"priority": 10, "sampling_params": {"max_new_tokens": 10000}},
                    {"priority": 9, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 8, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 7, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 6, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 5, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 0, "sampling_params": {"max_new_tokens": 100}},
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


def _verify_genereate_responses(
    responses: Tuple[int, Any, float],
    expected_code_and_error_message: Tuple[int, Any],
    e2e_latencies: List[Optional[float]],
):
    # Verify generate request responses match expected status codes and error messages
    for got, expected in zip(responses, expected_code_and_error_message):
        got_status, got_json = got
        expected_status, expected_err_msg = expected

        assert (
            got_status == expected_status
        ), f"expected_status:{expected_status}，actually:{got_status}"

        if got_status != 200:
            assert (
                got_json.get("object") == "error"
            ), f"expected object 'error', actually: {got_json.get('object')}"
            assert (
                got_json.get("message") == expected_err_msg
            ), f"expected_err_msg:{expected_err_msg}，actually: {got_json.get('message')}"
        else:
            assert "object" not in got_json
            assert "message" not in got_json

        if got_status == 200:
            e2e_latencies.append(got_json["meta_info"]["e2e_latency"])
        else:
            e2e_latencies.append(None)


def _verify_max_running_requests_and_max_queued_request_validation(
    max_running_requests: int, max_queued_requests: int
):
    # Verify server logs do not exceed max running/queued requests limits during test execution
    rr_pattern = re.compile(r"#running-req:\s*(\d+)")
    qr_pattern = re.compile(r"#queue-req:\s*(\d+)")

    if not os.path.exists(STDERR_FILENAME):
        return

    with open(STDERR_FILENAME, "r") as f:
        for line in f:
            rr_match = rr_pattern.search(line)
            if rr_match:
                assert (
                    int(rr_match.group(1)) <= max_running_requests
                ), f"running：{rr_match.group(1)} > {max_running_requests}"
            qr_match = qr_pattern.search(line)
            if qr_match:
                assert (
                    int(qr_match.group(1)) <= max_queued_requests
                ), f"queue：{qr_match.group(1)} > {max_queued_requests}"


if __name__ == "__main__":
    unittest.main()
