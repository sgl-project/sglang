"""
E2E tests for NPU pause/resume generation (in_place, retract, abort).

Covers:
  S1 : in_place concurrent pause blocks progress; resume completes all.
  S2 : single-request pause/resume (retract + in_place) with ignore_eos.
  S3 : retract multiple concurrent requests survive pause/resume.
  S4 : retract multiple cycles on same request; state not corrupted.
  S5 : abort mode aborts all in-flight requests; continue resumes service.
  S6 : pause/resume with empty running_batch (in_place + retract); engine stays operational.

"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=90, suite="full-1-npu-a3", nightly=True)


class TestNpuPauseResumeGeneration(CustomTestCase):
    """E2E: in_place / retract / abort mode for NPU pause/resume generation.

    [Test Category] RL Pause/Resume
    [Test Target] POST /pause_generation (mode=in_place, retract, abort),
                  POST /continue_generation
    """

    PAUSE_NUM_REQUESTS = 16
    PAUSE_MAX_NEW_TOKENS = 512
    PAUSE_DURATION = 5
    REQUEST_TIMEOUT = 180

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ── helpers ──────────────────────────────────────────────────

    def _generate(
        self, prompt_id: int, max_tokens: int = None, ignore_eos: bool = False
    ):
        """Send a single /generate request."""
        sampling_params: dict = {
            "temperature": 0.8,
            "max_new_tokens": max_tokens or self.PAUSE_MAX_NEW_TOKENS,
        }
        if ignore_eos:
            sampling_params["ignore_eos"] = True
        return requests.post(
            self.base_url + "/generate",
            json={
                "text": (
                    f"Question {prompt_id}: "
                    f"Write a short essay about the number {prompt_id}."
                ),
                "sampling_params": sampling_params,
            },
            timeout=self.REQUEST_TIMEOUT,
        )

    def _pause(self, mode: str = "in_place"):
        return requests.post(
            self.base_url + "/pause_generation",
            json={"mode": mode},
            timeout=30,
        )

    def _continue(self, torch_empty_cache: bool = True):
        return requests.post(
            self.base_url + "/continue_generation",
            json={"torch_empty_cache": torch_empty_cache},
            timeout=30,
        )

    def _assert_request_running(self, future, msg=None):
        """Assert a future is not yet done — request should still be in-flight."""
        self.assertFalse(
            future.done(),
            msg or "Request completed before pause — test precondition not met",
        )

    def _assert_finish_reason_length(self, body, msg=None):
        """Assert finish_reason is 'length' (ignore_eos=True path)."""
        self.assertEqual(
            body["meta_info"]["finish_reason"]["type"],
            "length",
            msg
            or f"Expected finish_reason='length', got {body['meta_info']['finish_reason']}",
        )

    def test_in_place_pause_blocks_progress(self):
        """
        S1: Verify that in_place pause blocks all in-flight requests
        and resume allows them to complete successfully.
        """
        with ThreadPoolExecutor(max_workers=self.PAUSE_NUM_REQUESTS) as executor:
            futures = {
                executor.submit(self._generate, i): i
                for i in range(self.PAUSE_NUM_REQUESTS)
            }

            time.sleep(1)

            resp = self._pause("in_place")
            self.assertEqual(resp.status_code, 200)

            try:
                # Poll until pause takes effect: no new requests complete.
                done_before = sum(1 for f in futures if f.done())
                poll_deadline = time.time() + 5.0
                while time.time() < poll_deadline:
                    current_done = sum(1 for f in futures if f.done())
                    if current_done == self.PAUSE_NUM_REQUESTS:
                        self.fail(
                            f"All {self.PAUSE_NUM_REQUESTS} requests finished "
                            "before pause stabilized — either pause arrived "
                            "too late or PAUSE_MAX_NEW_TOKENS is too small "
                            "for the model to keep requests in-flight long "
                            "enough."
                        )
                    if current_done == done_before:
                        break
                    done_before = current_done
                    time.sleep(0.1)
                else:
                    self.fail(
                        f"Pause did not stabilize within 5s — "
                        f"done_before={done_before}, "
                        f"requests still completing during poll. "
                        f"pause_generation may not be taking effect."
                    )

                # Wait through the pause window — no new requests should finish
                time.sleep(self.PAUSE_DURATION)
                done_after = sum(1 for f in futures if f.done())

                print(
                    f"[DEBUG] done_before={done_before}, done_after={done_after}, "
                    f"PAUSE_NUM_REQUESTS={self.PAUSE_NUM_REQUESTS}, "
                    f"PAUSE_DURATION={self.PAUSE_DURATION}s"
                )
                self.assertEqual(
                    done_after - done_before,
                    0,
                    f"{done_after - done_before} requests completed during pause "
                    f"({done_before} before, {done_after} after) — "
                    f"pause_generation was not respected by the scheduler.",
                )
            finally:
                self._continue().raise_for_status()

            # Verify all requests complete successfully (engine resumed)
            completed = 0
            errors = []
            continue_start = time.time()
            for future in as_completed(futures, timeout=self.REQUEST_TIMEOUT):
                prompt_id = futures[future]
                try:
                    resp = future.result()
                    if resp.status_code == 200:
                        body = resp.json()
                        self.assertIn("text", body)
                        self.assertGreater(len(body["text"]), 0)
                        completed += 1
                    else:
                        errors.append(f"Request {prompt_id}: status={resp.status_code}")
                except Exception as e:
                    errors.append(f"Request {prompt_id}: exception={e}")

        continue_elapsed = time.time() - continue_start
        print(
            f"[DEBUG] continue_to_complete={continue_elapsed:.2f}s, "
            f"completed={completed}, errors={len(errors)}"
        )

        self.assertEqual(
            completed + len(errors),
            self.PAUSE_NUM_REQUESTS,
            "Some requests did not resolve within timeout — likely hung during pause.",
        )
        self.assertEqual(
            completed,
            self.PAUSE_NUM_REQUESTS,
            f"Some requests failed: {completed}/{self.PAUSE_NUM_REQUESTS}."
            f" Errors: {errors}",
        )

    def test_pause_resume_single_request(self):
        """
        S2: A single long-generation request paused mid-generation
        survives pause/resume and completes with finish_reason='length'
        (ignore_eos=True). Covers both retract and in_place modes.
        """
        PAUSE_WINDOW = 5

        for mode in ("retract", "in_place"):
            with self.subTest(mode=mode):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._generate,
                        0,
                        512,
                        ignore_eos=True,
                    )
                    time.sleep(1.0)

                    self._assert_request_running(
                        future,
                        f"Request completed before {mode} — "
                        "pause would be a no-op, test is meaningless",
                    )

                    self._pause(mode).raise_for_status()

                    try:
                        time.sleep(PAUSE_WINDOW)
                        self._assert_request_running(
                            future,
                            f"Request should be paused in {mode}, not completed",
                        )
                    finally:
                        self._continue().raise_for_status()

                    resp = future.result(timeout=self.REQUEST_TIMEOUT)

                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertIn("text", body)
                self._assert_finish_reason_length(body)
                self.assertGreater(
                    len(body["text"]),
                    100,
                    f"Output too short after {mode}+resume: {len(body['text'])} chars",
                )
                print(
                    f"[{mode}_single] text ({len(body['text'])} chars):\n{body['text'][:500]}..."
                )

    def test_retract_resume_multiple_requests(self):
        """
        S3: Multiple concurrent requests all survive retract+resume.
        """
        NUM_REQUESTS = 4

        with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
            futures = {
                executor.submit(self._generate, i, 1024, ignore_eos=True): i
                for i in range(NUM_REQUESTS)
            }

            time.sleep(1.0)

            running = sum(1 for f in futures if not f.done())
            self.assertGreater(
                running,
                0,
                f"All {NUM_REQUESTS} requests completed before retract — "
                "retract would be a no-op, test is meaningless",
            )

            self._pause("retract").raise_for_status()
            self._continue().raise_for_status()

            completed = 0
            for future in as_completed(futures, timeout=self.REQUEST_TIMEOUT):
                resp = future.result()
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self._assert_finish_reason_length(body)
                self.assertGreater(len(body["text"]), 20)
                print(
                    f"[retract_multi req={futures[future]}] text ({len(body['text'])} chars):\n{body['text'][:200]}..."
                )
                completed += 1

        self.assertEqual(completed, NUM_REQUESTS)

    def test_retract_multiple_cycles(self):
        """
        S4: Multiple retract/resume cycles on the same request do not
        corrupt state or cause the request to abort.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._generate,
                0,
                1024,
                ignore_eos=True,
            )

            for cycle in range(3):
                time.sleep(1.0)
                self._assert_request_running(
                    future,
                    f"Request completed before retract cycle {cycle + 1} — "
                    "retract would be a no-op, test is meaningless",
                )
                self._pause("retract").raise_for_status()
                self._continue().raise_for_status()

            resp = future.result(timeout=self.REQUEST_TIMEOUT * 2)

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self._assert_finish_reason_length(
            body,
            f"Request should finish with 'length' after 3 retract cycles, "
            f"got finish_reason={body['meta_info']['finish_reason']}",
        )
        self.assertGreater(len(body["text"]), 100)
        print(
            f"[retract_cycles] text ({len(body['text'])} chars):\n{body['text'][:500]}..."
        )

    def test_abort_pause_and_continue(self):
        """
        S5: abort mode aborts all in-flight requests, then continue
        resets is_pause and allows new requests.
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    lambda i=i: requests.post(
                        self.base_url + "/generate",
                        json={
                            "text": f"Prompt {i}: Write about the number {i}.",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 256,
                                "ignore_eos": True,
                            },
                        },
                        timeout=self.REQUEST_TIMEOUT,
                    )
                )
                for i in range(4)
            ]

            time.sleep(1)

            self._pause("abort").raise_for_status()

            for future in futures:
                resp = future.result(timeout=10)
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertEqual(
                    body["meta_info"]["finish_reason"]["type"],
                    "abort",
                    "ignore_eos=True requests should be aborted mid-generation",
                )

            self._continue().raise_for_status()

        # Engine must serve new requests after continue
        resp = self._generate(0, max_tokens=32)
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.json()["text"]), 0)

    def test_pause_empty_running_batch(self):
        """
        S6: Pausing and resuming with no active requests does not crash
        the engine, and subsequent requests are processed normally.
        """
        for mode in ("in_place", "retract"):
            with self.subTest(mode=mode):
                resp = self._pause(mode)
                self.assertEqual(resp.status_code, 200)

                self._continue().raise_for_status()

                resp = self._generate(0, max_tokens=32)
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertIn("text", body)
                self.assertGreater(len(body["text"]), 0)


if __name__ == "__main__":
    unittest.main()
