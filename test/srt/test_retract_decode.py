import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestRetractDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)
        time.sleep(1)  # wait for mem check

        assert self.process.poll() is None, "Server crashed during test"


class TestRetractDecodeChunkCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--disable-radix-cache", "--chunked-prefill-size", 128],
            )

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)
        time.sleep(1)  # wait for mem check

        assert self.process.poll() is None, "Server crashed during test"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestAbortAllWithRetraction(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_TEST_RETRACT"] = "1"
        # Here's a small trick: in scheduler.py, when SGLANG_TEST_RETRACT is enabled,
        # retraction is triggered when the batch size reaches 10.
        # However, since SGLANG_TEST_RETRACT_NO_PREFILL_BS is set to 6, the remaining 4
        # requests will stay in the waiting queue.
        os.environ["SGLANG_TEST_RETRACT_NO_PREFILL_BS"] = "6"
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--max-running-requests", 16, "--schedule-policy", "random"],
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
                    "max_new_tokens": 4000,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def test_abort_all_with_retraction(self):
        num_requests = 32
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode) for _ in range(num_requests)]

            # ensure the decode has been started and retractions happen.
            time.sleep(8)

            requests.post(
                self.base_url + "/abort_request",
                json={
                    "abort_all": True,
                },
            )

            abort_in_queue_count = 0
            abort_in_queue_with_none_empty_text = 0

            for future in as_completed(futures):
                self.assertEqual(
                    future.result()["meta_info"]["finish_reason"]["type"], "abort"
                )
                if (
                    future.result()["meta_info"]["finish_reason"]["message"]
                    == "Abort in waiting queue"
                ):
                    abort_in_queue_count += 1
                    if len(future.result()["output_ids"]) > 0:
                        abort_in_queue_with_none_empty_text += 1
            assert abort_in_queue_count > 0
            assert abort_in_queue_with_none_empty_text > 0
            print("Finished test_abort_all_with_retraction")


if __name__ == "__main__":
    unittest.main()
