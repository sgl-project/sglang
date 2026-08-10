"""End-to-end test for dropping stale requests during a weight-update pause.

The version is caller-declared, so no weight update is needed: stamping different
ints on different requests and pausing with a threshold exercises the whole path.

    python -m unittest test.manual.entrypoints.http_server.test_pause_abort_stale_start_version -v
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MAX_NEW_TOKENS = 256
THRESHOLD = 7
STALE_VERSION = 5
FRESH_VERSION = 9


class TestPauseAbortStaleStartVersion(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--disable-cuda-graph"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, start_weight_version=None) -> dict:
        payload = {
            "text": "Write a very long essay about distributed systems.",
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
        }
        if start_weight_version is not None:
            payload["start_weight_version"] = start_weight_version
        r = requests.post(f"{self.base_url}/generate", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    def _pause(self, mode, threshold=None) -> requests.Response:
        body = {"mode": mode}
        if threshold is not None:
            body["abort_below_start_weight_version"] = threshold
        return requests.post(
            f"{self.base_url}/pause_generation", json=body, timeout=60
        )

    def _continue(self) -> requests.Response:
        return requests.post(
            f"{self.base_url}/continue_generation", json={}, timeout=60
        )

    def _wait_running(self, n, timeout=60) -> None:
        """Block until the scheduler reports this much work, instead of sleeping.

        A fixed sleep races the server: pausing before the requests are admitted
        silently tests nothing."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            loads = requests.get(
                f"{self.base_url}/v1/loads?include=core", timeout=10
            ).json()["loads"]
            # omit_defaults drops zero-valued fields from the snapshot.
            last = [load.get("num_running_reqs", 0) for load in loads]
            if any(r >= n for r in last):
                return
            time.sleep(0.2)
        raise AssertionError(f"timed out waiting for running>={n}, saw {last}")

    def _assert_aborted(self, result, label):
        self.assertEqual(
            result["meta_info"]["finish_reason"]["type"], "abort", label
        )

    def _assert_completed(self, result, label):
        self.assertIn(
            result["meta_info"]["finish_reason"]["type"], ("length", "stop"), label
        )
        self.assertEqual(
            result["meta_info"]["completion_tokens"], MAX_NEW_TOKENS, label
        )

    def _sweep_case(self, mode):
        pool = ThreadPoolExecutor(max_workers=4)
        try:
            stale = [pool.submit(self._generate, STALE_VERSION) for _ in range(2)]
            fresh = [pool.submit(self._generate, FRESH_VERSION) for _ in range(2)]
            self._wait_running(4)

            self.assertEqual(self._pause(mode, THRESHOLD).status_code, 200)
            self.assertEqual(self._continue().status_code, 200)

            stale_results = [f.result() for f in stale]
            fresh_results = [f.result() for f in fresh]
        finally:
            pool.shutdown(wait=False)
            try:
                self._continue()
            except requests.RequestException:
                pass

        # The pause was the only thing issued, so an aborted stale request can
        # only have been dropped by it.
        for r in stale_results:
            self._assert_aborted(r, f"{mode} v{STALE_VERSION}")
        for r in fresh_results:
            self._assert_completed(r, f"{mode} v{FRESH_VERSION}")

    def test_retract_sweep_drops_stale_keeps_fresh(self):
        """retract re-prefills survivors on resume, so the stale ones must go
        before that recompute rather than after."""
        self._sweep_case("retract")

    def test_in_place_sweep_drops_stale_keeps_fresh(self):
        """in_place keeps the KV, so the stale ones are finalized by the first
        forward pass after continue_generation."""
        self._sweep_case("in_place")

    def test_undeclared_requests_are_never_swept(self):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            undeclared = [pool.submit(self._generate) for _ in range(2)]
            self._wait_running(2)
            self.assertEqual(self._pause("retract", THRESHOLD).status_code, 200)
            self.assertEqual(self._continue().status_code, 200)
            results = [f.result() for f in undeclared]
        finally:
            pool.shutdown(wait=False)
            try:
                self._continue()
            except requests.RequestException:
                pass

        # An unknown version cannot be evaluated, so it must not be guessed at.
        for r in results:
            self._assert_completed(r, "undeclared")

    def test_echoes_declared_version_and_omits_it_otherwise(self):
        declared = self._generate(FRESH_VERSION)
        self.assertEqual(
            declared["meta_info"]["start_weight_version"], FRESH_VERSION
        )
        # Absent, not null: a response for a caller not using the feature is
        # unchanged.
        self.assertNotIn("start_weight_version", self._generate()["meta_info"])

    def test_abort_mode_rejects_the_threshold(self):
        self.assertNotEqual(self._pause("abort", THRESHOLD).status_code, 200)


if __name__ == "__main__":
    unittest.main()
