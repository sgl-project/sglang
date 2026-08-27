import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Safety timeout for all HTTP requests to prevent CI from hanging forever.
_REQUEST_TIMEOUT = 60


class AbortAllMixin:
    """Test /abort_request with abort_all=True.

    Server needs sufficient --max-running-requests.
    """

    abort_all_num_requests: int = 32
    abort_all_max_new_tokens: int = 16000
    abort_all_sleep: float = 2

    def test_abort_all(self):
        num_requests = self.abort_all_num_requests

        def run_decode():
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": self.abort_all_max_new_tokens,
                        "ignore_eos": True,
                    },
                },
                timeout=_REQUEST_TIMEOUT,
            )
            return response.json()

        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(run_decode) for _ in range(num_requests)]

            time.sleep(self.abort_all_sleep)

            requests.post(
                self.base_url + "/abort_request",
                json={"abort_all": True},
                timeout=10,
            ).raise_for_status()

            for future in as_completed(futures):
                result = future.result()
                self.assertEqual(result["meta_info"]["finish_reason"]["type"], "abort")

            self.assertIsNone(self.process.poll())


class WaitingTimeoutMixin:
    """Test waiting queue timeout.

    Server needs SGLANG_REQ_WAITING_TIMEOUT and --max-running-requests=1.
    """

    waiting_timeout_num_requests: int = 2
    waiting_timeout_max_new_tokens: int = 512

    def test_waiting_timeout(self):
        num_requests = self.waiting_timeout_num_requests

        def run_decode():
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Today is ",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": self.waiting_timeout_max_new_tokens,
                        "ignore_eos": True,
                    },
                },
                timeout=_REQUEST_TIMEOUT,
            )
            return response.json()

        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(run_decode) for _ in range(num_requests)]

            error_count = 0
            for future in as_completed(futures):
                result = future.result()
                if result.get("object") == "error":
                    error_count += 1
                    self.assertEqual(result["code"], 503)

            self.assertEqual(error_count, 1)
            self.assertIsNone(self.process.poll())


class RunningTimeoutTwoWaveMixin:
    """Test running timeout with a two-wave pattern.

    Sends two waves with different forward_entry_time so that timeouts are
    triggered in separate batches. Regression test for
    https://github.com/sgl-project/sglang/pull/18760

    Server needs SGLANG_REQ_RUNNING_TIMEOUT and sufficient --max-running-requests
    to hold both waves.
    """

    running_timeout_num_wave1: int = 8
    running_timeout_num_wave2: int = 8
    running_timeout_sleep: float = 3
    running_timeout_max_new_tokens: int = 1024

    def test_running_timeout_no_crash(self):
        num_wave1 = self.running_timeout_num_wave1
        num_wave2 = self.running_timeout_num_wave2

        def run_decode():
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Write a long story about a magical kingdom.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": self.running_timeout_max_new_tokens,
                        "ignore_eos": True,
                    },
                },
                timeout=_REQUEST_TIMEOUT,
            )
            return response.json()

        with ThreadPoolExecutor(num_wave1 + num_wave2) as executor:
            futures1 = [executor.submit(run_decode) for _ in range(num_wave1)]

            time.sleep(self.running_timeout_sleep)

            futures2 = [executor.submit(run_decode) for _ in range(num_wave2)]

            for future in as_completed(futures1 + futures2):
                result = future.result()
                if result.get("object") == "error":
                    self.assertEqual(result["code"], 503)

        self.assertIsNone(self.process.poll())
