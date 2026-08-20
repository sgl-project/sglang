# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression tests for LoRA usage-counter accounting (LoRARegistry).

Every request lifecycle must balance the registry's acquire/release exactly:
``unload_lora_adapter`` waits for the usage counter to reach zero, so a
leaked acquire makes the unload hang forever, and an extra release lets an
adapter unload while requests still use it. Each test drives one lifecycle
shape against a freshly loaded adapter and then asserts the unload completes
promptly (i.e. the counter returned to zero).

Lifecycles covered: parallel-sampling success (n>1), client disconnect
mid-generation (n=1 and n=4), /abort_request on queued and running requests,
tokenizer-side validation failure (oversized prompt), scheduler-side
BAD_REQUEST (invalid grammar), and streaming n>1 disconnect (whose abort
acks target the parent request ids).
"""

import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=480,
    stage="base-b",
    runner_config="1-gpu-large",
)

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "philschmid/code-llama-3-1-8b-text-to-sql-lora"
SENTINEL_NAME = "sentinel"  # stays loaded for the whole server lifetime
TEST_NAME = "lifecycle_test_adapter"  # loaded/unloaded once per test
PROMPT = "Write a SQL query that selects all columns from a table."
CONTEXT_LEN = 2048
UNLOAD_TIMEOUT = 60  # generous; a balanced unload returns in milliseconds


class TestLoRAUsageCounter(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            BASE_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-lora",
                "--lora-paths",
                f"{SENTINEL_NAME}={ADAPTER}",
                "--max-loras-per-batch",
                "2",
                "--max-running-requests",
                "1",
                "--context-length",
                str(CONTEXT_LEN),
                "--mem-fraction-static",
                "0.8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ---------- helpers ----------

    def _post(self, path, payload, timeout=120):
        return requests.post(f"{self.base_url}{path}", json=payload, timeout=timeout)

    def _gen_payload(self, n=1, max_new_tokens=64, rid=None, stream=False):
        payload = {
            "text": PROMPT,
            "lora_path": TEST_NAME,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "stream": stream,
        }
        if n > 1:
            payload["sampling_params"]["n"] = n
        if rid is not None:
            payload["rid"] = rid
        return payload

    def _load_test_adapter(self):
        resp = self._post(
            "/load_lora_adapter",
            {"lora_name": TEST_NAME, "lora_path": ADAPTER},
        )
        self.assertTrue(resp.json().get("success"), resp.text)

    def _assert_unload_completes(self, case: str, quiesce: float = 8.0):
        """Unload the test adapter and assert it does not hang.

        A leaked usage counter makes /unload_lora_adapter wait for a counter
        that never reaches zero; run the unload on a worker thread with a
        deadline so the test fails crisply instead of timing out the suite.
        """
        time.sleep(quiesce)  # let in-flight aborts/finishes drain
        result = {}

        def unload():
            resp = self._post(
                "/unload_lora_adapter", {"lora_name": TEST_NAME}, timeout=UNLOAD_TIMEOUT
            )
            result["status"] = resp.status_code
            result["body"] = resp.text

        start = time.monotonic()
        worker = threading.Thread(target=unload, daemon=True)
        worker.start()
        worker.join(UNLOAD_TIMEOUT)
        elapsed = time.monotonic() - start

        self.assertFalse(
            worker.is_alive(),
            f"[{case}] unload_lora_adapter still blocked after {UNLOAD_TIMEOUT}s "
            "— LoRA usage counter leaked (never returned to zero).",
        )
        self.assertEqual(result.get("status"), 200, f"[{case}] {result}")
        self.assertLess(
            elapsed,
            10.0,
            f"[{case}] unload took {elapsed:.1f}s; a balanced counter unloads "
            "immediately, a slow unload means the counter was nonzero.",
        )

    def _gen_in_thread(self, results, key, **kwargs):
        def run():
            try:
                results[key] = self._post(
                    "/generate", self._gen_payload(**kwargs), timeout=120
                ).json()
            except Exception as exc:  # disconnects are expected in some tests
                results[key] = repr(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    # ---------- lifecycle cases ----------

    def test_parallel_sampling_success(self):
        """n>1 success: the prefix-cache warm-up sub-request must not release."""
        self._load_test_adapter()
        resp = self._post("/generate", self._gen_payload(n=4, max_new_tokens=32))
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(len(resp.json()), 4)
        self._assert_unload_completes("n4-success", quiesce=2.0)

    def test_disconnect_mid_generation(self):
        """Client disconnects while generating (n=1 and n=4, non-streaming)."""
        self._load_test_adapter()
        for n in (1, 4):
            with self.assertRaises(requests.exceptions.RequestException):
                self._post(
                    "/generate",
                    self._gen_payload(n=n, max_new_tokens=1500),
                    timeout=3,  # hard-close the connection mid-generation
                )
        self._assert_unload_completes("disconnect-midgen", quiesce=15.0)

    def test_abort_request_queued_and_running(self):
        """/abort_request on a running and a queued request (1 running slot)."""
        self._load_test_adapter()
        results = {}
        running = self._gen_in_thread(
            results, "running", rid="usage-counter-running", max_new_tokens=1500
        )
        time.sleep(2)
        queued = self._gen_in_thread(
            results, "queued", rid="usage-counter-queued", max_new_tokens=1500
        )
        time.sleep(2)  # max-running-requests=1 -> second request is queued
        self._post("/abort_request", {"rid": "usage-counter-queued"})
        time.sleep(1)
        self._post("/abort_request", {"rid": "usage-counter-running"})
        running.join(60)
        queued.join(60)
        self._assert_unload_completes("abort-queued-running", quiesce=8.0)

    def test_validation_failure_after_acquire(self):
        """Oversized prompt: rejected after the LoRA counter was acquired."""
        self._load_test_adapter()
        resp = self._post(
            "/generate",
            {
                "input_ids": [10] * (CONTEXT_LEN + 100),
                "lora_path": TEST_NAME,
                "sampling_params": {"max_new_tokens": 8},
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self._assert_unload_completes("oversized-prompt", quiesce=2.0)

    def test_scheduler_bad_request(self):
        """Invalid grammar: scheduler-side BAD_REQUEST finish."""
        self._load_test_adapter()
        payload = self._gen_payload(max_new_tokens=16)
        payload["sampling_params"]["json_schema"] = '{"type":'  # unparseable
        resp = self._post("/generate", payload)
        self.assertEqual(resp.status_code, 400, resp.text)
        self._assert_unload_completes("scheduler-bad-request", quiesce=8.0)

    def test_streaming_parallel_disconnect(self):
        """Streaming n>1 disconnect: abort acks target the parent rids."""
        self._load_test_adapter()
        resp = requests.post(
            f"{self.base_url}/generate",
            json=self._gen_payload(n=4, max_new_tokens=100, stream=True),
            stream=True,
            timeout=30,
        )
        next(resp.iter_content(chunk_size=64))
        resp.close()  # create_abort_task fires ~2s later on the parent rids
        self._assert_unload_completes("stream-n4-disconnect", quiesce=25.0)


if __name__ == "__main__":
    unittest.main()
