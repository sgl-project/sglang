"""Tests for the drain primitive — block until in-flight requests complete
or a timeout elapses.

The in-flight test uses the HTTP server because Engine.drain and Engine.generate
both call self.loop.run_until_complete on the same event loop and cannot be
issued concurrently from two threads against a single Engine instance. The
HTTP path naturally handles that.
"""

import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, suite="stage-b-test-1-gpu-small")


class TestEngineDrain(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _drain(self, timeout):
        r = requests.post(
            self.base_url + "/drain",
            json={"timeout": timeout},
            timeout=timeout + 30 if timeout is not None else 60,
        )
        self.assertEqual(r.status_code, 200)
        return r.json()

    def test_drain_idle_returns_immediately(self):
        start = time.perf_counter()
        body = self._drain(timeout=5.0)
        elapsed = time.perf_counter() - start
        self.assertTrue(body["success"])
        self.assertEqual(body["remaining_requests"], 0)
        self.assertLess(elapsed, 1.0)

    def test_drain_with_inflight_request_times_out(self):
        result_holder = {}

        def submit_long_generate():
            try:
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": "Tell me a long story about a robot.",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 64,
                        },
                    },
                    timeout=120,
                )
                result_holder["status"] = r.status_code
            except Exception as e:
                result_holder["error"] = repr(e)

        t = threading.Thread(target=submit_long_generate)
        t.start()
        # Let the request reach the tokenizer manager and register a rid.
        time.sleep(0.5)

        try:
            body = self._drain(timeout=0.2)
            self.assertFalse(body["success"])
            self.assertGreater(body["remaining_requests"], 0)
        finally:
            t.join(timeout=120)
            self.assertFalse(t.is_alive(), "background generate did not finish")

        # After the request finishes, drain should report idle.
        body = self._drain(timeout=5.0)
        self.assertTrue(body["success"])
        self.assertEqual(body["remaining_requests"], 0)
        self.assertEqual(result_holder.get("status"), 200)

    def test_drain_no_timeout_completes_after_inflight_finishes(self):
        def submit_short_generate():
            requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
                timeout=60,
            )

        t = threading.Thread(target=submit_short_generate)
        t.start()
        time.sleep(0.2)

        body = self._drain(timeout=None)
        t.join(timeout=60)
        self.assertTrue(body["success"])
        self.assertEqual(body["remaining_requests"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
