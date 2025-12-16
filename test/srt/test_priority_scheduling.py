import asyncio
import os
import re
import types
import unittest
from typing import Any, List, Optional, Tuple
from unittest import mock

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.grpc.sglang_scheduler_pb2 import SamplingParams
from sglang.srt.managers import scheduler as scheduler_mod
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
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

    def test_priority_scheduling_preemption_token_offset_calculation(self):
        """
        Verify correct token offset calculation during preemption.

        This test specifically targets the bug where rem_total_token_offset was incorrectly
        calculated using the incoming request's tokens instead of the preempted request's tokens
        (related to issue #13111 and PR #13201).

        THE BUG:
        In schedule_policy.py line 700, the code was using:
            self.rem_total_token_offset -= self._get_running_request_total_token_offset(req)
        Instead of:
            self.rem_total_token_offset -= self._get_running_request_total_token_offset(running_req)

        WHY THIS TEST CATCHES THE BUG:
        - Request 1 (preempted): 8000 tokens - This is what SHOULD be freed
        - Request 3 (incoming):  1000 tokens - This is what WAS freed (bug)
        - Token difference: 8000 - 1000 = 7000 tokens incorrectly accounted

        With the bug, the system thinks it only freed 1000 tokens instead of 8000 tokens.
        This causes incorrect memory accounting and can lead to:
        1. Scheduler believes less memory is available than actually is
        2. Subsequent requests (like Request 4) may fail to schedule or cause issues
        3. Memory calculations become increasingly inaccurate with each preemption

        The test creates a scenario where:
        1. A low-priority request with many tokens (8000) starts running
        2. A high-priority request with few tokens (1000) arrives and triggers preemption
        3. The system must correctly free 8000 tokens from the preempted request
        4. Additional requests can be scheduled only if tokens were correctly freed
        5. Execution order validates priority-based scheduling works correctly

        The large token difference (8x) makes the bug's impact obvious and testable.
        """
        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "priority": 0,
                        "sampling_params": {"max_new_tokens": 8000},
                    },  # Low priority, large token count - will be preempted
                    {
                        "priority": 1,
                        "sampling_params": {"max_new_tokens": 5000},
                    },  # Medium priority, medium token count - queued initially
                    {
                        "priority": 100,
                        "sampling_params": {"max_new_tokens": 1000},
                    },  # High priority, small token count - triggers preemption
                    {
                        "priority": 50,
                        "sampling_params": {"max_new_tokens": 2000},
                    },  # Should be schedulable after correct token accounting
                ],
            )
        )

        # All requests should complete successfully
        # The key is that the fourth request should be schedulable because
        # the system correctly freed tokens from the first (preempted) request
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

        # Verify execution order: high priority requests finish before low priority ones
        # Request 3 (priority 100) should finish first
        # Request 4 (priority 50) should finish second
        # Request 2 (priority 1) should finish third
        # Request 1 (priority 0) should finish last (after being preempted)

        # FIXME(harrison lim)
        # assert e2e_latencies[2] < e2e_latencies[3] < e2e_latencies[1] < e2e_latencies[0]


class TestPrioritySchedulingPreemption(unittest.TestCase):
    def test_no_token_triggers_preempt_and_retry(self):
        """
        Simulate add_one_req returning NO_TOKEN first, then CONTINUE after
        preempt_to_schedule frees resources. The test verifies:
          - preempt_to_schedule is called
          - add_one_req is retried for the same request
          - the returned ScheduleBatch contains the request
        """

        # Minimal fake self used as 'self' when calling the unbound method
        fake_self = types.SimpleNamespace()

        # Set up scheduler-like attributes used by get_new_batch_prefill
        fake_self.schedule_enhancer = None
        fake_self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)

        dummy_req = Req(
            "r1",
            origin_input_text="test",
            origin_input_ids=[],
            sampling_params=SamplingParams(),
        )

        fake_self.waiting_queue = [dummy_req]
        fake_self.chunked_req = None
        fake_self.chunked_prefill_size = None
        fake_self.new_token_ratio = 1.0
        fake_self.max_prefill_tokens = 100
        fake_self.is_mixed_chunk = False
        fake_self.priority_scheduling_preemption_threshold = 0
        fake_self.server_args = types.SimpleNamespace(prefill_max_requests=1)
        fake_self.enable_lora = False
        fake_self.tp_worker = types.SimpleNamespace(can_run_lora_batch=lambda s: True)
        fake_self.enable_hierarchical_cache = False
        fake_self.enable_hicache_storage = False
        fake_self.tree_cache = None
        fake_self.req_to_token_pool = types.SimpleNamespace(available_size=lambda: 100)
        fake_self.page_size = 1
        fake_self.token_to_kv_pool_allocator = None

        fake_self.model_config = types.SimpleNamespace(is_matryoshka=lambda: False)
        fake_self.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
        fake_self.dllm_config = None
        fake_self.enable_metrics = False
        fake_self.policy = types.SimpleNamespace(calc_priority=lambda q: None)
        fake_self.grammar_queue = None
        fake_self.truncation_align_size = None
        fake_self.current_scheduler_metrics_enabled = lambda: False
        fake_self.enable_overlap = False

        # control how many allocatable reqs scheduler thinks it can accept
        fake_self.get_num_allocatable_reqs = lambda running_bs: 10

        # Use NULL disaggregation so PREFILL-specific branches are skipped
        fake_self.disaggregation_mode = DisaggregationMode.NULL

        fake_self.try_preemption = True

        created_adder = {}

        class FakeAdder:
            def __init__(self, *args, **kwargs):
                created_adder["instance"] = self
                self.can_run_list = []
                self.preempt_list = []
                self._calls = {}
                self.preempt_called = False
                self.new_chunked_req = None

            def add_one_req(self, req, **kwargs):
                # First call for this req -> NO_TOKEN; second call -> success (CONTINUE)
                key = id(req)
                self._calls[key] = self._calls.get(key, 0) + 1
                if self._calls[key] == 1:
                    return AddReqResult.NO_TOKEN
                self.can_run_list.append(req)
                return AddReqResult.CONTINUE

            def preempt_to_schedule(self, req, server_args):
                self.preempt_called = True
                # Simulate successful preemption freeing resources
                return True

        # Patch PrefillAdder used by get_new_batch_prefill to use our FakeAdder
        with mock.patch.object(scheduler_mod, "PrefillAdder", new=FakeAdder):
            # Patch ScheduleBatch.init_new to return a simple ScheduleBatch from can_run_list
            def fake_init_new(cls, can_run_list, *args, **kwargs):
                return ScheduleBatch(reqs=list(can_run_list))

            with mock.patch.object(
                scheduler_mod.ScheduleBatch, "init_new", new=classmethod(fake_init_new)
            ):
                with mock.patch.object(
                    scheduler_mod.ScheduleBatch,
                    "prepare_for_extend",
                    new=lambda self: None,
                ):  # suppress debug logging
                    # Execute the unbound method with our fake_self
                    result_batch = scheduler_mod.Scheduler.get_new_batch_prefill(
                        fake_self
                    )

        # Assertions
        self.assertIn("instance", created_adder, "FakeAdder was not created")
        adder = created_adder["instance"]
        self.assertTrue(
            getattr(adder, "preempt_called", False),
            "preempt_to_schedule was not called",
        )
        count = adder._calls.get(id(dummy_req), 0)
        self.assertEqual(
            count, 2, f"expected add_one_req to be called twice; got {count}"
        )
        self.assertIsInstance(result_batch, ScheduleBatch)
        self.assertEqual(len(result_batch.reqs), 1)
        self.assertIs(result_batch.reqs[0], dummy_req)


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
