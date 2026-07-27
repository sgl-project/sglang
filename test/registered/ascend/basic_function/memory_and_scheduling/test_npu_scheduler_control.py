import multiprocessing
import os
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.abort_timeout_kit import AbortAllMixin, WaitingTimeoutMixin
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    read_output,
)

# Mirror the temp-file names used by sglang.test.test_utils so that the
# streamed output thread can pick them up. We intentionally re-implement the
# body of run_and_check_memory_leak here because that helper hard-codes
# `DEFAULT_MODEL_NAME_FOR_TEST` ("meta-llama/Llama-3.1-8B-Instruct"), which is
# not available on ModelScope and thus cannot run on NPU CI runners.
_STDERR_FILENAME = "/tmp/stderr.txt"
_STDOUT_FILENAME = "/tmp/stdout.txt"


def _run_and_check_memory_leak_npu(
    workload_func,
    disable_radix_cache,
    enable_mixed_chunk,
    disable_overlap,
    chunked_prefill_size,
    assert_has_abort,
    api_key=None,
):
    """NPU-friendly replacement for run_and_check_memory_leak.

    Differs only in that it uses LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH (a
    ModelScope-resolvable path) instead of DEFAULT_MODEL_NAME_FOR_TEST, and
    adds NPU required args (--attention-backend ascend, --disable-cuda-graph).
    """
    other_args = [
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--log-level",
        "debug",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]
    if disable_radix_cache:
        other_args += ["--disable-radix-cache"]
    if enable_mixed_chunk:
        other_args += ["--enable-mixed-chunk"]
    if disable_overlap:
        other_args += ["--disable-overlap-schedule"]

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    port = random.randint(4000, 5000)
    base_url = f"http://127.0.0.1:{port}"

    # Create files and launch the server
    stdout = open(_STDOUT_FILENAME, "w")
    stderr = open(_STDERR_FILENAME, "w")
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        return_stdout_stderr=(stdout, stderr),
        api_key=api_key,
    )

    # Launch a thread to stream the output
    output_lines = []
    t = threading.Thread(target=read_output, args=(output_lines, _STDERR_FILENAME))
    t.start()

    # Run the workload
    workload_func(base_url, model)

    # Clean up everything
    kill_process_tree(process.pid)
    stdout.close()
    stderr.close()
    if os.path.exists(_STDOUT_FILENAME):
        os.remove(_STDOUT_FILENAME)
    if os.path.exists(_STDERR_FILENAME):
        os.remove(_STDERR_FILENAME)
    kill_process_tree(process.pid)
    t.join()

    # Assert success
    has_new_server = False
    has_leak = False
    has_abort = False
    for line in output_lines:
        if "Uvicorn running" in line:
            has_new_server = True
        if "leak" in line:
            has_leak = True
        if "Abort" in line:
            has_abort = True

    assert has_new_server
    assert not has_leak
    if assert_has_abort:
        assert has_abort


register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


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
        _run_and_check_memory_leak_npu(
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
        _run_and_check_memory_leak_npu(
            lambda base_url, model: self.workload_func(base_url, model, api_key),
            disable_radix_cache=False,
            enable_mixed_chunk=False,
            disable_overlap=False,
            chunked_prefill_size=8192,
            assert_has_abort=True,
            api_key=api_key,
        )


class TestSchedulerControl(AbortAllMixin, PauseResumeInPlaceMixin, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-running-requests",
                "8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
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
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
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
                    "16",
                    "--schedule-policy",
                    "random",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
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
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0.001):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--max-running-requests=1",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestAbortWithRunningTimeout(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_REQ_RUNNING_TIMEOUT.override(0.001),
            envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.override(False),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--skip-server-warmup",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                ],
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
