"""Per-commit streaming-session tests.

Default config + EagleV2RetractLargePage + abort-leak repro stay per-commit.
Other variants (Retract / Eagle / EagleV2 / EagleRetractLargePage) live in
test_streaming_session_extra.py.
"""

import concurrent.futures
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.streaming_session_kit import (
    AbortLeakReproKitMixin,
    StreamingSessionKitMixin,
)
from sglang.test.server_fixtures.streaming_session_fixture import (
    ABORT_REPRO_CHUNKED_PREFILL_SIZE,
    ABORT_REPRO_CONTEXT_LEN,
    ABORT_REPRO_PAGE_SIZE,
    StreamingSessionServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=811, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=811, suite="stage-b-test-1-gpu-large-amd")


class TestStreamingSession(StreamingSessionServerBase, StreamingSessionKitMixin):
    """Default streaming-session config (small model, no spec)."""

    extra_args = ["--chunked-prefill-size", "512"]


class TestStreamingSessionEagleV2RetractLargePage(TestStreamingSession):
    """EAGLE3 spec v2 + retract + page=256."""

    model = DEFAULT_TARGET_MODEL_EAGLE3
    extra_args = [
        "--chunked-prefill-size",
        "4096",
        "--dtype=float16",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model",
        DEFAULT_DRAFT_MODEL_EAGLE3,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "256",
    ]
    env_overrides = [
        ("SGLANG_TEST_RETRACT", True),
        ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True),
    ]


class TestStreamingSessionAbortLeakRepro(
    StreamingSessionServerBase, AbortLeakReproKitMixin
):
    extra_args = [
        "--chunked-prefill-size",
        str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
        "--context-length",
        str(ABORT_REPRO_CONTEXT_LEN),
        "--page-size",
        str(ABORT_REPRO_PAGE_SIZE),
        "--max-running-requests",
        "32",
        "--log-level",
        "info",
    ]


class QueuedSessionTurnMixin:
    """Helpers for putting a session turn in the waiting queue and watching it.

    A capped running-request budget plus a blocker request that fills it is what
    makes queue placement deterministic, and the scheduler's own gauges say when
    each step has actually landed, so no step is timed.
    """

    BLOCKER_RID = "queued-session-blocker"
    BLOCKER_TOKENS = 512
    WAIT_S = 60

    def _gauge_total(self, name):
        """Read a scheduler gauge's total series.

        Under priority scheduling the same gauge also carries a per-priority
        breakdown; the total is the series with the default `priority=""`.
        """
        response = requests.get(self.base_url + "/metrics", timeout=10)
        self.assertEqual(response.status_code, 200, response.text)
        for line in response.text.splitlines():
            if not line.startswith(name + "{"):
                continue
            labels, value = line[len(name) + 1 :].rsplit("}", 1)
            if 'priority="' in labels and 'priority=""' not in labels:
                continue
            return float(value)
        self.fail(f"metric {name} missing from /metrics")

    def _running_reqs(self):
        return self._gauge_total("sglang:num_running_reqs")

    def _queued_reqs(self):
        return self._gauge_total("sglang:num_queue_reqs")

    def _generate(
        self, input_ids, session_id, rid=None, max_new_tokens=8, priority=None
    ):
        payload = {
            "input_ids": input_ids,
            "session_params": {"id": session_id},
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        }
        if rid is not None:
            payload["rid"] = rid
        if priority is not None:
            payload["priority"] = priority
        return requests.post(self.base_url + "/generate", json=payload, timeout=180)

    def _abort(self, rid):
        return requests.post(
            self.base_url + "/abort_request", json={"rid": rid}, timeout=30
        )

    def _run_blocker(self, priority=None):
        payload = {
            "text": "Count slowly starting from one:",
            "rid": self.BLOCKER_RID,
            "sampling_params": {
                "temperature": 0,
                # Long enough to hold the slot across the removal below, short
                # enough to retire on its own: removing a *running* request is
                # not what these tests cover.
                "max_new_tokens": self.BLOCKER_TOKENS,
                # Without this the blocker stops at EOS, frees the slot, and
                # the session turn is admitted instead of queued.
                "ignore_eos": True,
            },
        }
        if priority is not None:
            payload["priority"] = priority
        return requests.post(self.base_url + "/generate", json=payload, timeout=180)

    def _await_gauge(self, read, expected, what):
        deadline = time.time() + self.WAIT_S
        observed = None
        while time.time() < deadline:
            observed = read()
            if observed == expected:
                return
            time.sleep(0.05)
        self.fail(f"timed out waiting for {what}; observed={observed}")

    def _await_running(self, expected, what):
        self._await_gauge(self._running_reqs, expected, what)

    def _await_queued(self, expected, what):
        self._await_gauge(self._queued_reqs, expected, what)

    def _open_session(self):
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "streaming": True},
        ).json()
        self.addCleanup(
            requests.post,
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        return session_id


class TestStreamingSessionQueuedAbort(
    StreamingSessionServerBase, QueuedSessionTurnMixin
):
    """Aborting a session turn while it is still queued must not wedge the session.

    The scheduler confirms the queue placement itself: only its waiting-queue
    removal path reports "Abort in waiting queue".
    """

    # A held SessionSlot keeps one req-pool slot between turns, so the cap must
    # leave one more for the blocker; `--decode-log-interval 1` refreshes
    # num_running_reqs every decode step (the default is too stale to sync on).
    extra_args = [
        "--max-running-requests",
        "2",
        "--enable-metrics",
        "--decode-log-interval",
        "1",
    ]

    TARGET_RID = "queued-abort-target"

    def test_queued_session_turn_abort_keeps_session_usable(self):
        session_id = self._open_session()

        turn1_ids = self.tokenizer.encode("The capital of France is")
        aborted_ids = self.tokenizer.encode(" Ignore this sentence entirely.")[1:]
        turn3_ids = self.tokenizer.encode(" Now say something else.")[1:]

        # Turn 1 is served normally and becomes the session's committed prefix.
        turn1 = self._generate(turn1_ids, session_id)
        self.assertEqual(turn1.status_code, 200, turn1.text)
        turn1_meta = turn1.json()["meta_info"]
        committed_len = turn1_meta["prompt_tokens"] + turn1_meta["completion_tokens"]

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            pool.submit(self._run_blocker)
            # Once the blocker is running it holds the only slot, so the turn
            # submitted next can only sit in the waiting queue.
            self._await_running(1, "the blocker to start running")

            target = pool.submit(
                self._generate, aborted_ids, session_id, self.TARGET_RID
            )
            # /generate only returns once the turn is done, so retry the abort
            # until it lands. The turn cannot progress while the slot is held.
            deadline = time.time() + self.WAIT_S
            while not target.done() and time.time() < deadline:
                self.assertEqual(self._abort(self.TARGET_RID).status_code, 200)
                time.sleep(0.1)
            self.assertTrue(target.done(), "queued session turn was never aborted")
            target_body = target.result(timeout=self.WAIT_S).json()
        finally:
            # The blocker retires on its own; just join it.
            pool.shutdown(wait=True)

        finish_reason = target_body["meta_info"]["finish_reason"]
        self.assertEqual(finish_reason["type"], "abort", target_body["meta_info"])
        # Only Scheduler.abort_request's waiting-queue branch reports this, so
        # it is the server's own confirmation that the turn never started.
        self.assertEqual(finish_reason["message"], "Abort in waiting queue")
        self.assertEqual(target_body["output_ids"], [])

        self._await_running(0, "the server to drain")

        # The session must still accept turns.
        turn3 = self._generate(turn3_ids, session_id)
        self.assertEqual(turn3.status_code, 200, turn3.text)
        turn3_meta = turn3.json()["meta_info"]

        turn3_finish = turn3_meta.get("finish_reason") or {}
        self.assertNotEqual(
            turn3_finish.get("type"),
            "abort",
            f"follow-up turn was rejected: {turn3_finish.get('message')!r}",
        )

        # Distinguishes both wrong fixes: discarding the committed prefix gives
        # len(turn3_ids), and committing the aborted prompt adds len(aborted_ids).
        self.assertEqual(
            turn3_meta["prompt_tokens"],
            committed_len + len(turn3_ids),
            "turn 3 context is not exactly the committed prefix plus its own prompt",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200, "server unhealthy after queued abort")


class TestStreamingSessionPriorityEviction(
    StreamingSessionServerBase, QueuedSessionTurnMixin
):
    """A session turn evicted by the queue limit must not wedge the session.

    `_abort_on_queued_limit` is the other way a turn leaves the waiting queue
    without ever running: with priority scheduling on and the queue at its cap,
    an arriving higher-priority request evicts the queued lower-priority one.
    That removal is as final as an explicit abort, so it needs the same session
    finalization.
    """

    # `--max-queued-requests 1` means the queue holds exactly the session turn,
    # so the arriving request must displace it rather than queue behind it.
    # The priorities below stay within
    # `--priority-scheduling-preemption-threshold` (default 10) so the running
    # blocker is never preempted: this exercises the queue limit, nothing else.
    extra_args = [
        "--max-running-requests",
        "2",
        "--max-queued-requests",
        "1",
        "--enable-priority-scheduling",
        "--default-priority-value",
        "1",
        "--enable-metrics",
        "--decode-log-interval",
        "1",
    ]

    EVICTED_RID = "priority-eviction-target"
    PREEMPTOR_RID = "priority-eviction-preemptor"
    LOW_PRIORITY = 1
    HIGH_PRIORITY = 5

    def _run_preemptor(self):
        """The arrival that finds the queue full and outranks what is in it."""
        return requests.post(
            self.base_url + "/generate",
            json={
                "text": "Say hello:",
                "rid": self.PREEMPTOR_RID,
                "priority": self.HIGH_PRIORITY,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=180,
        )

    def test_priority_evicted_session_turn_keeps_session_usable(self):
        session_id = self._open_session()

        turn1_ids = self.tokenizer.encode("The capital of France is")
        evicted_ids = self.tokenizer.encode(" Ignore this sentence entirely.")[1:]
        turn3_ids = self.tokenizer.encode(" Now say something else.")[1:]

        # Turn 1 is served normally and becomes the session's committed prefix.
        turn1 = self._generate(turn1_ids, session_id, priority=self.LOW_PRIORITY)
        self.assertEqual(turn1.status_code, 200, turn1.text)
        turn1_meta = turn1.json()["meta_info"]
        committed_len = turn1_meta["prompt_tokens"] + turn1_meta["completion_tokens"]

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        try:
            pool.submit(self._run_blocker, self.HIGH_PRIORITY)
            # Once the blocker is running it holds the only slot, so the turn
            # submitted next can only sit in the waiting queue.
            self._await_running(1, "the blocker to start running")

            evicted = pool.submit(
                self._generate,
                evicted_ids,
                session_id,
                self.EVICTED_RID,
                priority=self.LOW_PRIORITY,
            )
            # Wait for the turn to actually reach the queue, so the request sent
            # next finds the queue full and at a lower priority than itself.
            # This only sequences the two submissions; what proves the eviction
            # path ran is the abort message the scheduler sends back below.
            self._await_queued(1, "the session turn to reach the waiting queue")

            preemptor = pool.submit(self._run_preemptor)
            evicted_response = evicted.result(timeout=self.WAIT_S)
            self.assertEqual(preemptor.result(timeout=self.WAIT_S).status_code, 200)
        finally:
            # The blocker retires on its own; just join it.
            pool.shutdown(wait=True)

        # Unlike an explicit abort, the queue limit aborts with
        # SERVICE_UNAVAILABLE, so the turn comes back as an HTTP error instead
        # of a finished generation -- it produced no tokens at all.
        self.assertEqual(evicted_response.status_code, 503, evicted_response.text)
        # Only the priority branch of _abort_on_queued_limit reports this, so it
        # is the server's own confirmation of which path removed the turn. A
        # plain queue-limit rejection would say "The request queue is full."
        self.assertEqual(
            evicted_response.json()["message"],
            "The request is aborted by a higher priority request.",
        )

        self._await_running(0, "the server to drain")

        # The session must still accept turns, i.e. it is not left in-flight.
        turn3 = self._generate(turn3_ids, session_id, priority=self.LOW_PRIORITY)
        self.assertEqual(turn3.status_code, 200, turn3.text)
        turn3_meta = turn3.json()["meta_info"]

        turn3_finish = turn3_meta.get("finish_reason") or {}
        self.assertNotEqual(
            turn3_finish.get("type"),
            "abort",
            f"follow-up turn was rejected: {turn3_finish.get('message')!r}",
        )

        # Distinguishes both wrong fixes: discarding the committed prefix gives
        # len(turn3_ids), and committing the evicted prompt adds len(evicted_ids).
        self.assertEqual(
            turn3_meta["prompt_tokens"],
            committed_len + len(turn3_ids),
            "turn 3 context is not exactly the committed prefix plus its own prompt",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code, 200, "server unhealthy after priority eviction"
        )


if __name__ == "__main__":
    unittest.main()
