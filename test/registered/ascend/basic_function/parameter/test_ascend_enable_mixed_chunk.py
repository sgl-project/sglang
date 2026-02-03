import unittest
import requests
import time
import threading
from datetime import datetime

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from typing import Dict,Any
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

CONFIG = {
    "TARGET_TOKEN_COUNT": 2500,
    "CHUNK_SIZE": 1024,
    "REQUEST_COUNT": 50,
    "TIMEOUT": 600
}

FINAL_STATISTICS: Dict[str, Dict[str, Any]] = {
    "mixed_enabled": {},
    "mixed_disabled": {}
}

def build_long_input_text_for_token():
    """Construct long input text with enough tokens (common function for consistent input)"""
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (CONFIG["TARGET_TOKEN_COUNT"] // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

def send_generate_request(task_id, request_results):
    # record single request elapsed time
    single_start_time = time.time()
    
    long_input_text = build_long_input_text_for_token()
    
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "text": long_input_text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 32,
            },
        },
        timeout=CONFIG["TIMEOUT"]
    )
    
    single_end_time = time.time()
    single_elapsed_time = round(single_end_time - single_start_time, 4)

    request_results.append({
        "task_id": task_id,
        "status_code": response.status_code,
        "single_elapsed_time": single_elapsed_time
    })
    
    print(f"[Task {task_id}] Request completed, status code: {response.status_code}, elapsed time: {single_elapsed_time} seconds")

def start_server(with_mixed: bool):
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chunked-prefill-size", str(CONFIG["CHUNK_SIZE"])
    ]
    
    # Add --enable-mixed-chunk parameter if needed
    if with_mixed:
        other_args.insert(0, "--enable-mixed-chunk")
    
    # Start server
    process = popen_launch_server(
        LLAMA_3_2_1B_WEIGHTS_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )

    return process

def calculate_statistics(request_results):
    """Common statistics function: calculate average, max, min elapsed time (remove overall elapsed time)"""
    success_requests = [r for r in request_results if r["status_code"] == 200]
    if not success_requests:
        return None
    
    # Extract elapsed time of successful requests
    elapsed_times = [r["single_elapsed_time"] for r in success_requests]
    
    # Calculate statistical indicators
    return {
        "success_count": len(success_requests),
        "total_count": len(request_results),
        "avg_elapsed": round(sum(elapsed_times) / len(elapsed_times), 4),
        "max_elapsed": round(max(elapsed_times), 4),
        "min_elapsed": round(min(elapsed_times), 4)
    }

class TestMixedChunkEnabled(CustomTestCase):
    """Testcase: Verify the baseline performance of disabled --enable-mixed-chunk with 50 concurrent long-token requests (TARGET_TOKEN_COUNT=2500).

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk;
    """
    @classmethod
    def setUpClass(cls):
        """Start server (with mixed chunk enabled)"""
        print("=== Starting Server (--enable-mixed-chunk ENABLED) ===")
        cls.process = start_server(with_mixed=True)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(10)

    def test_mixed_chunk_with_multi_requests(self):
        #Multi-request test , collect statistics
        request_results = []
        
        # Print test start information
        print(f"\n=== [Mixed Chunk ENABLED] Test started, timestamp: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== Starting {CONFIG['REQUEST_COUNT']} request threads to create queue scenario ===")
        
        # Create and start multiple threads
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Calculate statistics
        statistics = calculate_statistics(request_results)
        
        FINAL_STATISTICS["mixed_enabled"] = {
            "detail": statistics
        }
        
        print(f"  Average elapsed time per request: {statistics['avg_elapsed']} seconds")
        print(f"  Maximum elapsed time per request: {statistics['max_elapsed']} seconds")
        print(f"  Minimum elapsed time per request: {statistics['min_elapsed']} seconds")

        # Add assertGreater after both tests are completed (verify optimization effect)
        self._run_performance_assertions()

    def _run_performance_assertions(self):
        # verify performance optimization
        print("=== Running Performance Assertions ===")
        
        enabled = FINAL_STATISTICS["mixed_enabled"]
        disabled = FINAL_STATISTICS["mixed_disabled"]

        # Extract core statistical data
        enabled_avg = enabled["detail"]["avg_elapsed"]
        disabled_avg = disabled["detail"]["avg_elapsed"]
        
        avg_optimize_rate = round(((disabled_avg - enabled_avg) / disabled_avg) * 100, 2)
        
        # Print assertion preparation information
        print(f"Average Elapsed Time Comparison")
        print(f"   Mixed Enabled: {enabled_avg}s | Mixed Disabled: {disabled_avg}s | Optimization Rate: {avg_optimize_rate}%")
        
        # Core assertion: Average elapsed time (disabled > enabled)
        self.assertGreater(disabled_avg, enabled_avg, 
                           f"Assertion Failed: Average elapsed time - Mixed Disabled ({disabled_avg}s) is not greater than Mixed Enabled ({enabled_avg}s)")

class TestMixedChunkDisabled(CustomTestCase):
    """Testcase: Verify the baseline performance of disabled --enable-mixed-chunk with 50 concurrent long-token requests (TARGET_TOKEN_COUNT=2500).

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk;
    """

    @classmethod
    def setUpClass(cls):
        # Start server (with mixed chunk disabled)
        print("\n" + "="*60)
        print("=== Starting Server (--enable-mixed-chunk DISABLED) ===")
        cls.process = start_server(with_mixed=False)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(10)

    def test_mixed_chunk_with_multi_requests(self):
        request_results = []
        
        # Print test start information
        print(f"\n=== [Mixed Chunk DISABLED] Test started, timestamp: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== Starting {CONFIG['REQUEST_COUNT']} request threads to create queue scenario ===")
        
        # Create and start multiple threads
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Calculate statistics
        statistics = calculate_statistics(request_results)
        
        # Store final statistics
        FINAL_STATISTICS["mixed_disabled"] = {
            "detail": statistics
        }
        
        # Print current test result
        print(f"  Average elapsed time per request: {statistics['avg_elapsed']} seconds")
        print(f"  Maximum elapsed time per request: {statistics['max_elapsed']} seconds")
        print(f"  Minimum elapsed time per request: {statistics['min_elapsed']} seconds")


if __name__ == "__main__":
    # Execute all test cases with detailed output
    unittest.main(verbosity=2, exit=False)
