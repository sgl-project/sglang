import asyncio
import os
import re
import time
import unittest
from typing import Any, List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
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


class TestPrioritySchedulingPreemptionThreshold(CustomTestCase): 
    """Testcase: Verify the priority scheduling preemption threshold mechanism and execution order by controlling the priority and sequence of sending requests..

    [Test Category] Parameter
    [Test Target] --priority-scheduling-preemption-threshold;--enable-priority-scheduling;--max-running-requests;--max-queued-requests
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
            other_args = [
                "--max-running-requests", "1",  # Limit concurrent running requests to 1
                "--max-queued-requests", "10",
                "--enable-priority-scheduling", # Enable priority scheduling (required for preemption)
                "--priority-scheduling-preemption-threshold", "5",
                "--disable-cuda-graph",
                "--attention-backend", "ascend",
                "--tp-size", "1",
                "--mem-fraction-static", "0.8",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
    
    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        
        _verify_running_queued_requests(1, 10)
        
        cls.stdout.close()
        cls.stderr.close()
        if os.path.exists(STDOUT_FILENAME):
            os.remove(STDOUT_FILENAME)
        if os.path.exists(STDERR_FILENAME):
            os.remove(STDERR_FILENAME)
    
    def test_preemption_threshold_execution_order(self):
        """Test core preemption threshold logic (priority difference ≥ 5 triggers preemption / execution priority)
        
        Test Scenario:
        1. Request A (priority=2, long-running: 2000 tokens) - starts first, occupies running slot
        2. After 0.5s, Request C (priority=10, short: 100 tokens) - priority difference (10-2=8 ≥5) → preempts/executes first
        3. After another 0.5s, Request B (priority=5, short: 100 tokens) - priority difference (5-2=3 <5) → no preemption, executes after C
        Expected Execution Order: C → B → A (Expected Latency: C < B < A)
        """

        time.sleep(5)
        request_a = {
            "text":"repeat the words France",
            "priority": 2,
            "sampling_params": {"max_new_tokens": 2000}  
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        task_a = loop.create_task(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_a]
            )
        )
        
        loop.run_until_complete(asyncio.sleep(0.5))
        
        request_c = {
            "text":"repeat the words France",
            "priority": 10,
            "sampling_params": {"max_new_tokens": 100}
        }
        responses_c = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_c]
            )
        )
        
        loop.run_until_complete(asyncio.sleep(0.5))
        request_b = {
            "text":"repeat the words France",
            "priority": 5,
            "sampling_params": {"max_new_tokens": 100} 
        }
        responses_b = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_b]
            )
        )
        
        responses_a = loop.run_until_complete(task_a)
        
        all_responses = responses_a + responses_c + responses_b
        
        loop.close()
        
        expected_status = [(200, None)] * 3  
        e2e_latencies = []
        _verify_generate_responses(all_responses, expected_status, e2e_latencies)
        
        latency_a = e2e_latencies[0]
        latency_c = e2e_latencies[1]
        latency_b = e2e_latencies[2]
        
        assert latency_c < latency_b < latency_a, \
            f"expected C<B<A，actually：C={latency_c}, A={latency_a}, B={latency_b}"

    def test_preemption_threshold_execution_order_exa(self):
        """Test extended preemption threshold logic (same low priority → FIFO execution order)
        
        Test Scenario:
        1. Request A (priority=2, long-running: 2000 tokens) - starts first, occupies running slot
        2. After 0.5s, Request C (priority=10, short: 100 tokens) - priority difference ≥5 → executes first
        3. After another 0.5s, Request B (priority=2, long-running: 2000 tokens) - same priority as A → FIFO order (A before B)
        Expected Execution Order: C → A → B (Expected Latency: C < A < B)
        """

        time.sleep(5)
        request_a = {
            "text":"repeat the words France",
            "priority": 2,
            "sampling_params": {"max_new_tokens": 2000} 
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        task_a = loop.create_task(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_a]
            )
        )
        
        loop.run_until_complete(asyncio.sleep(0.5))
        
        request_c = {
            "text":"repeat the words France",
            "priority": 10,
            "sampling_params": {"max_new_tokens": 100}  
        }
        responses_c = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_c]
            )
        )
        loop.run_until_complete(asyncio.sleep(0.5))
        request_b = {
            "text":"repeat the words France",
            "priority": 2,
            "sampling_params": {"max_new_tokens": 2000} 
        }
        responses_b = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_b]
            )
        )
        
        responses_a = loop.run_until_complete(task_a)
  
        all_responses = responses_a + responses_c + responses_b
        
        loop.close()
        
        expected_status = [(200, None)] * 3 
        e2e_latencies = []
        _verify_generate_responses(all_responses, expected_status, e2e_latencies)
        
        latency_a = e2e_latencies[0]
        latency_c = e2e_latencies[1]
        latency_b = e2e_latencies[2]
        
        assert latency_c < latency_a < latency_b, \
            f"expected C<A<B，accurate：C={latency_c}, A={latency_a}, B={latency_b}"

def _verify_generate_responses(
    responses: Tuple[int, Any, float],
    expected_code_and_error: Tuple[int, Any],
    e2e_latencies: List[Optional[float]],
):
    # Verify generate request responses match expected status codes and extract valid e2e latencies (fix syntax/logic errors)
    e2e_latencies.clear() 
    for got, expected in zip(responses, expected_code_and_error):
        got_status, got_json = got
        expected_status, expected_err = expected
        
        assert got_status == expected_status, \
            f"expected_status:{expected_status}，actually{got_status}，response：{got_json}"
        
        if got_status == 200:
            assert "error" not in got_json, f"error：{got_json.get('error', 'unknown error')}"
            
            assert "meta_info" in got_json, "response does not have the necessary 'meta_info'"
            assert "e2e_latency" in got_json["meta_info"], "response does not have the necessary 'e2e_latency'"
            e2e_latencies.append(got_json["meta_info"]["e2e_latency"])
        else:
            e2e_latencies.append(None)

def _verify_running_queued_requests(
    max_running_requests: int, max_queued_requests: int
):
    # Verify server logs do not exceed max running/queued requests limits during test execution (Ascend log compatible)
    rr_pattern = re.compile(r"#running-req:\s*(\d+)")
    qr_pattern = re.compile(r"#queue-req:\s*(\d+)")
    
    if not os.path.exists(STDERR_FILENAME):
        return
    
    with open(STDERR_FILENAME, "r", encoding="utf-8") as f:
        for line in f:
            rr_match = rr_pattern.search(line)
            if rr_match:
                running_req_count = int(rr_match.group(1))
                assert running_req_count <= max_running_requests, \
                    f"running_req_count：{running_req_count} > {max_running_requests}"
            
            qr_match = qr_pattern.search(line)
            if qr_match:
                queued_req_count = int(qr_match.group(1))
                assert queued_req_count <= max_queued_requests, \
                    f"queued_req_count：{queued_req_count} > {max_queued_requests}"

if __name__ == "__main__":
    unittest.main()
