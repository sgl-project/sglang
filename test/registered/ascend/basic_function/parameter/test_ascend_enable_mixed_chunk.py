import unittest
import requests
import time
import threading
from datetime import datetime

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from typing import Dict, Any
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
        return {
            "success_count": 0,
            "total_count": len(request_results),
            "avg_elapsed": 0.0,
            "max_elapsed": 0.0,
            "min_elapsed": 0.0
        }
    
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

class TestMixedChunkPerformanceComparison(CustomTestCase):
    """Testcase: Compare performance between --enable-mixed-chunk ON/OFF with 50 concurrent long-token requests.

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk;
    """

    test_results: Dict[str, Dict[str, Any]] = {
        "mixed_enabled": {},
        "mixed_disabled": {}
    }

    @classmethod
    def _run_concurrent_tests(cls, with_mixed: bool) -> Dict[str, Any]:
        process = start_server(with_mixed=with_mixed)
        time.sleep(5) 

        request_results = []
        mixed_status = "ENABLED" if with_mixed else "DISABLED"
        print(f"\n=== [Mixed Chunk {mixed_status}] Test started, timestamp: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== Starting {CONFIG['REQUEST_COUNT']} request threads to create queue scenario ===")
        
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        

        statistics = calculate_statistics(request_results)

        kill_process_tree(process.pid)
        time.sleep(10) 
  
        print(f"\n=== [Mixed Chunk {mixed_status}] Test Results ===")
        print(f"  Success count / Total count: {statistics['success_count']} / {statistics['total_count']}")
        print(f"  Average elapsed time per request: {statistics['avg_elapsed']} seconds")
        print(f"  Maximum elapsed time per request: {statistics['max_elapsed']} seconds")
        print(f"  Minimum elapsed time per request: {statistics['min_elapsed']} seconds")
        
        return statistics

    def test_1_mixed_chunk_disabled(self):
        statistics = self._run_concurrent_tests(with_mixed=False)
        self.__class__.test_results["mixed_disabled"] = {"detail": statistics}


    def test_2_mixed_chunk_enabled(self):
        statistics = self._run_concurrent_tests(with_mixed=True)
        self.__class__.test_results["mixed_enabled"] = {"detail": statistics}
        
        # Add assertGreater after both tests are completed (verify optimization effect)
        self._run_performance_assertions()

    def _run_performance_assertions(self):
        # verify performance optimization
        print("\n" + "="*60)
        print("=== Running Performance Assertions ===")
        
        enabled = self.__class__.test_results["mixed_enabled"]["detail"]
        disabled = self.__class__.test_results["mixed_disabled"]["detail"]

        # Extract core statistical data
        enabled_avg = enabled["avg_elapsed"]
        disabled_avg = disabled["avg_elapsed"]
        
        self.assertNotEqual(disabled_avg, 0.0, "Disabled mode average elapsed time is 0, invalid data")
        self.assertNotEqual(enabled_avg, 0.0, "Enabled mode average elapsed time is 0, invalid data")
    
        avg_optimize_rate = round(((disabled_avg - enabled_avg) / disabled_avg) * 100, 2)
        
        print(f"\nAverage Elapsed Time Comparison")
        print(f"   Mixed Enabled: {enabled_avg}s | Mixed Disabled: {disabled_avg}s | Optimization Rate: {avg_optimize_rate}%")

        # Core assertion: Average elapsed time (disabled > enabled)
        self.assertGreater(disabled_avg, enabled_avg, 
                           f"Assertion Failed: Average elapsed time - Mixed Disabled ({disabled_avg}s) is not greater than Mixed Enabled ({enabled_avg}s)")

if __name__ == "__main__":
    # Execute all test cases with detailed output
    unittest.main(verbosity=2, exit=False)
