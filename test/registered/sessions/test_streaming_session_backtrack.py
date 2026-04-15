"""
Streaming session backtracking tests: offset-based backtrack, KV reuse,
and memory leak verification.

All tests share a single server (DEFAULT_SMALL_MODEL) with streaming sessions
and chunked prefill enabled.

Usage:
    python -m pytest test_streaming_session_backtrack.py -xvs
    python -m unittest test_streaming_session_backtrack.TestStreamingSessionBacktrack
"""

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

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-large")


class TestStreamingSessionBacktrack(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-streaming-session",
                "--chunked-prefill-size",
                "512",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_session(self):
        requests.post(self.base_url + "/flush_cache")
        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 4096, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _close_session(self, session_id):
        resp = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(resp.status_code, 200)

    def _generate(self, text, session_params=None, max_new_tokens=16):
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "no_stop_trim": True,
            },
        }
        if session_params is not None:
            payload["session_params"] = session_params
        resp = requests.post(self.base_url + "/generate", json=payload, timeout=120)
        self.assertEqual(resp.status_code, 200, f"Generate failed: {resp.text}")
        return resp.json()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_basic_backtrack_with_offset(self):
        """Backtrack via offset to an earlier position; KV should be reused."""
        sid = self._open_session()

        # Turn 1
        r1 = self._generate("Hello world", session_params={"id": sid})
        turn1_end = (
            r1["meta_info"]["prompt_tokens"] + r1["meta_info"]["completion_tokens"]
        )

        # Turn 2 — appends to turn 1
        r2 = self._generate(" How are you?", session_params={"id": sid})
        self.assertGreater(
            r2["meta_info"]["cached_tokens"], 0, "Turn 2 should reuse KV"
        )

        # Backtrack to end of turn 1 via offset — should still reuse KV
        r3 = self._generate(
            " What about France?",
            session_params={"id": sid, "offset": turn1_end},
        )
        self.assertEqual(
            r3["meta_info"]["cached_tokens"],
            turn1_end,
            "Backtrack via offset should reuse exactly turn1's KV",
        )

        self._close_session(sid)

    def test_offset_truncates_context(self):
        """Offset truncates the base context to the specified position."""
        sid = self._open_session()

        # Turn 1
        r1 = self._generate("Hello world", session_params={"id": sid})

        # Turn 2 — normal append
        r2 = self._generate(" Continue.", session_params={"id": sid})

        # Turn 3 — backtrack with a small offset
        r3 = self._generate(
            " Continue.",
            session_params={"id": sid, "offset": 5},
        )
        append_tokens = (
            r2["meta_info"]["prompt_tokens"]
            - r1["meta_info"]["prompt_tokens"]
            - r1["meta_info"]["completion_tokens"]
        )
        self.assertEqual(
            r3["meta_info"]["prompt_tokens"],
            5 + append_tokens,
            "prompt_tokens should be exactly offset + new input tokens",
        )

        self._close_session(sid)

    def test_no_memory_leak(self):
        """Multiple backtracks via offset should not leak KV memory."""
        sid = self._open_session()

        # Turn 1
        r1 = self._generate("Hello world", session_params={"id": sid})
        turn1_end = (
            r1["meta_info"]["prompt_tokens"] + r1["meta_info"]["completion_tokens"]
        )

        # Several forward-then-backtrack cycles
        for i in range(5):
            self._generate(
                f" Turn {i}.",
                session_params={"id": sid, "offset": turn1_end},
            )

        self._close_session(sid)
        requests.post(self.base_url + "/flush_cache")
        time.sleep(3)

        health = requests.get(self.base_url + "/health")
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after repeated backtracks — likely a KV memory leak.",
        )

        # After close + flush, a fresh session should start with 0 cached tokens.
        # If KV was leaked, the old tokens would still occupy slots.
        sid2 = self._open_session()
        r = self._generate("Fresh start", session_params={"id": sid2})
        self.assertEqual(
            r["meta_info"]["cached_tokens"],
            0,
            "Fresh session after flush should have 0 cached tokens — KV may have leaked.",
        )
        self._close_session(sid2)


if __name__ == "__main__":
    unittest.main()
