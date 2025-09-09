import json
import multiprocessing
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_and_check_memory_leak,
)


class TestAbort(CustomTestCase):
    def workload_func(self, base_url, model):
        def process_func():
            def run_one(_):
                prompt = """
                System: You are a helpful assistant.
                User: What is the capital of France?
                Assistant: The capital of France is
                """

                response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 2048,
                        },
                    },
                )
                ret = response.json()

            with ThreadPoolExecutor(16) as executor:
                list(executor.map(run_one, list(range(16))))

        p = multiprocessing.Process(target=process_func)
        p.start()
        time.sleep(0.5)
        p.terminate()
        time.sleep(10)

    def test_memory_leak(self):
        run_and_check_memory_leak(
            self.workload_func,
            disable_radix_cache=False,
            enable_mixed_chunk=False,
            disable_overlap=False,
            chunked_prefill_size=8192,
            assert_has_abort=True,
        )


class TestAbortAll(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--max-running-requests", 8],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16000,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def test_abort_all(self):
        num_requests = 32
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode) for _ in range(num_requests)]

            # ensure the decode has been started
            time.sleep(2)

            requests.post(
                self.base_url + "/abort_request",
                json={
                    "abort_all": True,
                },
            )

            for future in as_completed(futures):
                self.assertEqual(
                    future.result()["meta_info"]["finish_reason"]["type"], "abort"
                )


if __name__ == "__main__":
    unittest.main()
