"""
Integration test for abort_request functionality with a SGLang server.

Run with:
    python -m unittest sglang.test.srt.entrypoints.http_server.test_abort_request -v
"""

import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAbortRequest(CustomTestCase):
    """Integration test class for abort request functionality."""

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        """Launch the server."""
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--disable-cuda-graph"],
        )

        cls.completion_url = f"{cls.base_url}/generate"
        cls.abort_url = f"{cls.base_url}/abort_request"
        cls.health_url = f"{cls.base_url}/health"

        print(f"Server started at {cls.base_url}")

    @classmethod
    def tearDownClass(cls):
        """Clean up the server."""
        kill_process_tree(cls.process.pid)

    def _send_completion_request(
        self,
        text: str,
        request_id: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        stream: bool = True,
    ) -> requests.Response:
        """Send a completion request to the server."""
        payload = {
            "text": text,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
            "stream": stream,
            "rid": request_id,
        }

        response = requests.post(
            self.completion_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
            stream=stream,
        )

        return response

    def _send_abort_request(self, request_id: str) -> requests.Response:
        """Send an abort request."""
        payload = {"rid": request_id}
        return requests.post(self.abort_url, json=payload, timeout=10)

    def _check_server_health(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def test_abort_during_non_streaming_generation(self):
        """Test aborting a non-streaming request during generation."""
        self.assertTrue(self._check_server_health(), "Server should be healthy")

        request_id = "test_abort_non_streaming"
        completion_result = {}

        def run_completion():
            response = self._send_completion_request(
                "Write a detailed essay about artificial intelligence",
                max_tokens=500,
                temperature=1,
                request_id=request_id,
                stream=False,
            )

            if response.status_code == 200:
                result = response.json()
                completion_result["text"] = result.get("text", "")
                completion_result["finish_reason"] = result.get("meta_info", {}).get(
                    "finish_reason"
                )

        completion_thread = threading.Thread(target=run_completion)
        completion_thread.start()
        time.sleep(0.1)

        abort_response = self._send_abort_request(request_id)
        completion_thread.join()

        self.assertEqual(abort_response.status_code, 200)
        self.assertIsNotNone(completion_result, "Should have completion result")
        if completion_result:
            finish_reason_obj = completion_result.get("finish_reason")
            self.assertIsNotNone(finish_reason_obj, "Should have finish_reason")
            if finish_reason_obj:
                self.assertEqual(
                    finish_reason_obj.get("type"), "abort", "Should be aborted"
                )

    def test_batch_requests_with_selective_abort(self):
        """Test multiple concurrent requests with selective abort of one request."""
        self.assertTrue(self._check_server_health(), "Server should be healthy")

        request_ids = ["batch_test_0", "batch_test_1", "batch_test_2"]
        abort_target_id = "batch_test_1"
        completion_results = {}
        threads = []

        def run_completion(req_id, prompt):
            response = self._send_completion_request(
                f"Write a story about {prompt}",
                max_tokens=100,
                temperature=0.8,
                request_id=req_id,
                stream=False,
            )

            if response.status_code == 200:
                result = response.json()
                completion_results[req_id] = {
                    "text": result.get("text", ""),
                    "finish_reason": result.get("meta_info", {}).get("finish_reason"),
                }

        # Start all requests
        prompts = ["a knight's adventure", "a space discovery", "a chef's restaurant"]
        for i, req_id in enumerate(request_ids):
            thread = threading.Thread(target=run_completion, args=(req_id, prompts[i]))
            threads.append(thread)
            thread.start()

        # Abort one request
        time.sleep(0.1)
        abort_response = self._send_abort_request(abort_target_id)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Verify results
        self.assertEqual(abort_response.status_code, 200)

        # Check aborted request
        aborted_result = completion_results.get(abort_target_id)
        self.assertIsNotNone(
            aborted_result, f"Aborted request {abort_target_id} should have result"
        )
        if aborted_result:
            aborted_finish_reason = aborted_result.get("finish_reason")
            self.assertIsNotNone(
                aborted_finish_reason, "Aborted request should have finish_reason"
            )
            if aborted_finish_reason:
                self.assertEqual(aborted_finish_reason.get("type"), "abort")

        # Check other requests completed normally
        normal_completions = 0
        for req_id in request_ids:
            if req_id != abort_target_id and req_id in completion_results:
                result = completion_results[req_id]
                if result:
                    finish_reason = result.get("finish_reason")
                    if finish_reason and finish_reason.get("type") == "length":
                        normal_completions += 1

        self.assertEqual(
            normal_completions, 2, "Other 2 requests should complete normally"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
