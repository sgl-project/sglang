import multiprocessing
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.abort_timeout_kit import AbortAllMixin, WaitingTimeoutMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_and_check_memory_leak,
)

register_cuda_ci(est_time=131, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd")


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


class TestAbortWithApiKey(CustomTestCase):
    def workload_func(self, base_url, model, api_key: str):
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
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.json()

            with ThreadPoolExecutor(16) as executor:
                list(executor.map(run_one, list(range(16))))

        p = multiprocessing.Process(target=process_func)
        p.start()
        time.sleep(0.5)
        p.terminate()
        time.sleep(10)

    def test_memory_leak_with_api_key(self):
        api_key = "test-api-key"
        run_and_check_memory_leak(
            lambda base_url, model: self.workload_func(base_url, model, api_key),
            disable_radix_cache=False,
            enable_mixed_chunk=False,
            disable_overlap=False,
            chunked_prefill_size=8192,
            assert_has_abort=True,
            api_key=api_key,
        )


class TestAbortAll(AbortAllMixin, CustomTestCase):
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

    def _generate_with_rid(self, rid, max_new_tokens=8):
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "rid": rid,
            },
            timeout=30,
        )

    def test_duplicate_rid_sequential_ok(self):
        rid = "dup-rid-test-sequential"
        resp1 = self._generate_with_rid(rid)
        self.assertEqual(resp1.status_code, 200)
        self.assertNotIn("error", resp1.json())

        resp2 = self._generate_with_rid(rid)
        self.assertEqual(resp2.status_code, 200)
        self.assertNotIn("error", resp2.json())

    def test_duplicate_rid_concurrent_rejected(self):
        rid = "dup-rid-test-concurrent"
        results = {}

        def send(key, max_tokens):
            results[key] = self._generate_with_rid(rid, max_new_tokens=max_tokens)

        t1 = threading.Thread(target=send, args=("first", 512))
        t2 = threading.Thread(target=send, args=("second", 8))
        t1.start()
        time.sleep(0.1)
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        r1, r2 = results["first"], results["second"]
        self.assertTrue(
            r1.status_code == 400 or r2.status_code == 400,
            "One of the concurrent duplicate-rid requests should be rejected",
        )

        rejected = r2 if r2.status_code == 400 else r1
        self.assertIn("Duplicate request ID", rejected.json()["error"]["message"])

    def test_duplicate_rid_in_batch(self):
        rid = "dup-rid-batch"
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": ["Hello", "World"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "rid": [rid, rid],
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Duplicate request ID", response.json()["error"]["message"])

    def test_server_healthy_after_duplicate_rid(self):
        requests.post(
            f"{self.base_url}/generate",
            json={
                "text": ["Hello", "World"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "rid": ["dup-health", "dup-health"],
            },
            timeout=30,
        )

        resp = requests.get(f"{self.base_url}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)

        resp = self._generate_with_rid("after-dup-health")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text", resp.json())


class TestAbortAllWithRetraction(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Here's a small trick: in scheduler.py, when SGLANG_TEST_RETRACT is enabled,
        # retraction is triggered when the batch size reaches 10.
        # However, since SGLANG_TEST_RETRACT_NO_PREFILL_BS is set to 6, the remaining 4
        # requests will stay in the waiting queue.
        with (
            envs.SGLANG_TEST_RETRACT.override(True),
            envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.override(6),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--max-running-requests",
                    16,
                    "--schedule-policy",
                    "random",
                ],
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
                "return_logprob": True,
                "top_logprobs_num": 3,
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
            abort_in_queue_with_partial_gen = 0

            for future in as_completed(futures):
                result = future.result()
                meta_info = result["meta_info"]
                finish_reason = meta_info.get("finish_reason", {})

                self.assertEqual(finish_reason.get("type"), "abort")

                if finish_reason.get("message") == "Abort in waiting queue":
                    abort_in_queue_count += 1
                    output_ids = result.get("output_ids", [])

                    if len(output_ids) > 0:
                        abort_in_queue_with_partial_gen += 1

                        self.assertEqual(
                            meta_info.get("completion_tokens"), len(output_ids)
                        )
                        self.assertGreater(len(result.get("text", "")), 0)
                        self.assertIsNotNone(meta_info.get("weight_version"))
                        self.assertGreater(meta_info.get("e2e_latency"), 0)
                        for logprob_key in [
                            "output_token_logprobs",
                            "output_top_logprobs",
                        ]:
                            self.assertEqual(
                                len(meta_info.get(logprob_key, [])),
                                len(output_ids),
                                f"Length of '{logprob_key}' should match output_ids length",
                            )

            self.assertGreater(abort_in_queue_count, 0)
            self.assertGreater(abort_in_queue_with_partial_gen, 0)
            print("Finished test_abort_all_with_retraction")


class TestAbortWithWaitingTimeout(WaitingTimeoutMixin, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0.001):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--max-running-requests=1",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestAbortWithRunningTimeout(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(
            0.001
        ), envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.override(False):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--skip-server-warmup"],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_running_timeout(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Today is ",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 512,
                    "ignore_eos": True,
                },
            },
        )
        result = response.json()
        self.assertEqual(result["object"], "error")
        self.assertEqual(result["code"], 503)


if __name__ == "__main__":
    unittest.main()
