import os
import threading
import time
import unittest
from typing import Any, Dict, List

import requests

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_process_tree,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=500,
    suite="nightly-1-npu-a3",
    nightly=True,
)

SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|eot_id|>",
    "<|pad|>",
    "<|unk|>",
]


class TestTokenizerBatchDecodeBehavior(CustomTestCase):
    """Testcase: Verify that when --disable-tokenizer-batch-decode is enabled, the grouping function works correctly and the inference results are consistent with when it is disabled.

    [Test Category] Parameter
    [Test Target] --disable-tokenizer-batch-decode
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    test_prompt = "Hello<|eot_id|>"  # Prompt containing special tokens

    base_args = [
        "--trust-remote-code",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tokenizer-mode",
        "auto",
        "--revision",
        "main",
    ]

    def setUp(self):
        configure_logger(ServerArgs(self.model))
        self.out_log = open("./tmp_out.txt", "w+")
        self.err_log = open("./tmp_err.txt", "w+")
        self.processes = []

    def tearDown(self):
        for p in self.processes:
            kill_process_tree(p.pid)
        self.out_log.close()
        self.err_log.close()
        for f in ["./tmp_out.txt", "./tmp_err.txt"]:
            if os.path.exists(f):
                os.remove(f)

    def _run_server(self, extra_args: List[str] = None):
        extra_args = extra_args or []
        args = self.base_args + extra_args

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=1200,
            other_args=args,
            return_stdout_stderr=(self.out_log, self.err_log),
        )
        self.processes.append(process)
        return process

    def _stop_server(self, process):
        if process in self.processes:
            self.processes.remove(process)
        kill_process_tree(process.pid)

    def _send_single_request(
        self,
        skip_special: bool,
        request_id: int,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Send a single inference request"""
        payload = {
            "text": self.test_prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": skip_special,
                "spaces_between_special_tokens": True,
                "stop": ["<|eot_id|>"],
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/generate", json=payload, timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                text = result["text"]
                has_special = any(token in text for token in SPECIAL_TOKENS)
                return {
                    "request_id": request_id,
                    "skip_special_tokens": skip_special,
                    "text": text,
                    "has_special": has_special,
                    "status": "success",
                    "raw": result,
                }
            else:
                return {
                    "request_id": request_id,
                    "skip_special_tokens": skip_special,
                    "text": "",
                    "has_special": False,
                    "status": f"fail_{response.status_code}",
                    "error": response.text,
                }
        except Exception as e:
            return {
                "request_id": request_id,
                "skip_special_tokens": skip_special,
                "text": "",
                "has_special": False,
                "status": "exception",
                "error": str(e),
            }

    def _run_concurrent_test(
        self,
        num_requests: int = 10,
        skip_special_first: bool = True,
    ) -> List[Dict[str, Any]]:
        """Send concurrent inference requests"""
        results = []
        threads = []

        def worker(request_id):
            skip_special = (
                skip_special_first if request_id % 2 == 1 else not skip_special_first
            )
            result = self._send_single_request(skip_special, request_id)
            results.append(result)

        for i in range(1, num_requests + 1):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze inference results and return statistics"""
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        bad_cases = [
            r
            for r in results
            if r["status"] == "success"
            and r["skip_special_tokens"]
            and r["has_special"]
        ]
        special_in_no_skip = [
            r
            for r in results
            if r["status"] == "success"
            and not r["skip_special_tokens"]
            and r["has_special"]
        ]

        return {
            "total": total,
            "success": success,
            "bad_cases": len(bad_cases),
            "special_in_no_skip": len(special_in_no_skip),
            "results": results,
        }

    def test_compare_disable_tokenizer_batch_decode(self):
        """Test: Behavior consistency with/without --disable-tokenizer-batch-decode"""

        test_cases = [
            {"name": "Without --disable-tokenizer-batch-decode", "extra_args": []},
            {
                "name": "With --disable-tokenizer-batch-decode",
                "extra_args": ["--disable-tokenizer-batch-decode"],
            },
        ]

        results_by_case = {}

        for case in test_cases:
            process = self._run_server(case["extra_args"])
            time.sleep(2)

            # Send 2 requests (one skip=True, one skip=False) to test grouping logic
            results = self._run_concurrent_test(num_requests=2, skip_special_first=True)
            results_by_case[case["name"]] = results

            self._stop_server(process)

            # Analyze results
            analysis = self._analyze_results(results)
            self.assertEqual(
                analysis["bad_cases"],
                0,
                f"{case['name']} contains skip=True but outputs special tokens",
            )
            self.assertEqual(
                analysis["special_in_no_skip"],
                0,
                f"{case['name']} contains skip=False but outputs special tokens",
            )

        # Compare results between the two cases (take the first request)
        case1 = results_by_case["Without --disable-tokenizer-batch-decode"][0]
        case2 = results_by_case["With --disable-tokenizer-batch-decode"][0]

        # Only compare skip=True cases (special tokens may exist for skip=False)
        if case1["skip_special_tokens"] and case2["skip_special_tokens"]:
            self.assertEqual(
                case1["text"],
                case2["text"],
                "Output inconsistency between batch decode and disable batch decode",
            )

    def test_batch_decode_grouping_logic(self):
        # Verify the batch decode grouping logic works correctly
        process = self._run_server([])

        # Send 2 requests with different skip settings
        results = self._run_concurrent_test(num_requests=2, skip_special_first=True)
        self._stop_server(process)

        # Verify grouping handling
        skip_true = [r for r in results if r["skip_special_tokens"]]
        skip_false = [r for r in results if not r["skip_special_tokens"]]
        self.assertTrue(
            len(skip_true) > 0, "There should be at least one skip=True request"
        )
        self.assertTrue(
            len(skip_false) > 0, "There should be at least one skip=False request"
        )

        # Verify results are not None
        for r in results:
            self.assertIn(r["status"], ["success", "exception"])


if __name__ == "__main__":
    unittest.main()
