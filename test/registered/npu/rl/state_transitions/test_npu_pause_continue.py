"""P0 NPU tests: pause/continue generation around RL weight updates.

The scheduler must drain or freeze in-flight requests without emitting tokens
from a half-updated model, and must resume the same requests afterwards.
"""

import os
import threading
import time
import unittest

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    begin_weight_update,
    checkpoint_tensor_index,
    continue_generation,
    end_weight_update,
    generate,
    generation_signature,
    load_checkpoint_tensor,
    npu_available,
    npu_rl_env,
    npu_server,
    pause_generation,
    start_stream,
    update_weights_from_tensor,
    wait_for_first_token,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=900, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=900, suite="nightly-1-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
SERVER_ARGS = (
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-overlap-schedule",
    "--mem-fraction-static",
    "0.45",
    "--context-length",
    "1024",
    "--max-running-requests",
    "8",
)
QUEUE_ARGS = SERVER_ARGS[:-1] + ("1",)
SHORT_PROMPT = PROMPTS[0]
NORM_WEIGHT = "model.norm.weight"


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuPauseContinue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._server = npu_server(MODEL, extra_args=SERVER_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()
        cls.baseline = generation_signature(cls.base_url, PROMPTS, max_new_tokens=16)
        cls.tensor_index = checkpoint_tensor_index(MODEL)

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    def tearDown(self):
        try:
            continue_generation(self.base_url)
        except Exception:
            pass

    def _decode_async(self, max_new_tokens, results):
        def _run():
            try:
                results["response"] = generate(
                    self.base_url,
                    SHORT_PROMPT,
                    max_new_tokens=max_new_tokens,
                    timeout=300,
                )
            except Exception as exc:  # pragma: no cover - surfaced in the test
                results["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread

    def test_pause_abort_returns_inflight(self):
        request = start_stream(self.base_url, SHORT_PROMPT, max_new_tokens=512)
        wait_for_first_token(request, label="pause(abort)")

        pause_generation(self.base_url, mode="abort")
        request.thread.join(timeout=60)

        self.assertFalse(request.errors, f"stream failed: {request.errors}")
        self.assertFalse(
            request.thread.is_alive(), "request stream did not end after abort"
        )
        last = request.events[-1]
        finish_type = last.get("meta_info", {}).get("finish_reason", {}).get("type")
        self.assertEqual(
            finish_type, "abort", f"pause(abort) did not abort the request: {last}"
        )
        self.assertLess(
            len(last.get("output_ids", [])),
            512,
            "aborted request produced the full completion",
        )

        continue_generation(self.base_url)
        self.assertTrue(generate(self.base_url, SHORT_PROMPT, max_new_tokens=4))

    def test_pause_retract_recomputes_kv(self):
        request = start_stream(self.base_url, SHORT_PROMPT, max_new_tokens=64)
        wait_for_first_token(request, label="pause(retract)")

        pause_generation(self.base_url, mode="retract")
        time.sleep(0.2)
        continue_generation(self.base_url)
        request.thread.join(timeout=60)

        self.assertFalse(request.errors, f"stream failed: {request.errors}")
        # Native /generate streaming returns the cumulative output_ids per chunk.
        output_ids = list(request.events[-1].get("output_ids", []))
        self.assertEqual(
            len(output_ids),
            64,
            "retract/continue did not complete the full decode",
        )
        self.assertEqual(
            tuple(output_ids[: len(self.baseline[0])]),
            self.baseline[0],
            "retract/continue changed a greedy rollout",
        )

    def test_pause_in_place_freezes_output(self):
        request = start_stream(self.base_url, SHORT_PROMPT, max_new_tokens=64)
        wait_for_first_token(request, label="pause(in_place)")

        pause_generation(self.base_url, mode="in_place")
        time.sleep(0.5)
        frozen_tokens = len(request.events[-1].get("output_ids", []))
        self.assertGreater(frozen_tokens, 0, "in-place pause froze before any token")

        time.sleep(1.0)
        self.assertEqual(
            len(request.events[-1].get("output_ids", [])),
            frozen_tokens,
            "in-flight request kept emitting tokens while generation was paused",
        )

        continue_generation(self.base_url)
        request.thread.join(timeout=60)

        self.assertFalse(request.errors, f"stream failed: {request.errors}")
        self.assertFalse(
            request.thread.is_alive(), "request did not finish after continue"
        )
        output_ids = list(request.events[-1].get("output_ids", []))
        self.assertEqual(
            len(output_ids), 64, "in-place pause/continue did not complete the decode"
        )
        self.assertEqual(
            tuple(output_ids[: len(self.baseline[0])]),
            self.baseline[0],
            "in-place pause/continue changed a greedy rollout",
        )

    def test_pause_then_update_then_continue(self):
        original = load_checkpoint_tensor(self.tensor_index, NORM_WEIGHT)
        updated = original.new_full(original.shape, 0.5)

        request = start_stream(self.base_url, SHORT_PROMPT, max_new_tokens=64)
        wait_for_first_token(request, label="pause(update)")

        pause_generation(self.base_url, mode="in_place")
        time.sleep(0.5)
        frozen_tokens = len(request.events[-1].get("output_ids", []))
        self.assertGreater(frozen_tokens, 0, "in-place pause froze before any token")

        try:
            begin_weight_update(self.base_url)
            update_weights_from_tensor(
                self.base_url,
                [(NORM_WEIGHT, updated.to("npu"))],
                weight_version="paused-v1",
            )
            end_weight_update(self.base_url)

            time.sleep(0.5)
            self.assertEqual(
                len(request.events[-1].get("output_ids", [])),
                frozen_tokens,
                "in-flight request emitted tokens during the weight update",
            )

            continue_generation(self.base_url)
            request.thread.join(timeout=60)

            self.assertFalse(request.errors, f"stream failed: {request.errors}")
            self.assertFalse(
                request.thread.is_alive(),
                "frozen request did not finish after continue",
            )
            output_ids = list(request.events[-1].get("output_ids", []))
            self.assertEqual(
                len(output_ids),
                64,
                "frozen request did not complete after pause/update/continue",
            )

            response = generate(self.base_url, SHORT_PROMPT, max_new_tokens=8)
            self.assertEqual(
                response["meta_info"]["weight_version"],
                "paused-v1",
                "resumed request was not served by the updated weights",
            )
        finally:
            begin_weight_update(self.base_url)
            update_weights_from_tensor(
                self.base_url,
                [(NORM_WEIGHT, original.to("npu"))],
                weight_version="restored",
            )
            end_weight_update(self.base_url)

    def test_continue_without_pause_is_noop(self):
        continue_generation(self.base_url)
        self.assertTrue(generate(self.base_url, SHORT_PROMPT, max_new_tokens=4))

    def test_pause_twice_keeps_server_healthy(self):
        pause_generation(self.base_url, mode="abort")
        second = requests.post(
            self.base_url + "/pause_generation",
            json={"mode": "abort"},
            timeout=120,
        )
        self.assertEqual(second.status_code, 200, second.text)
        continue_generation(self.base_url)
        self.assertTrue(generate(self.base_url, SHORT_PROMPT, max_new_tokens=4))


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuPauseContinueQueued(CustomTestCase):
    """Pause behavior with one running request and one queued request."""

    @classmethod
    def setUpClass(cls):
        cls._server = npu_server(MODEL, extra_args=QUEUE_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()
        cls.baseline = generation_signature(cls.base_url, PROMPTS, max_new_tokens=16)

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    def tearDown(self):
        try:
            continue_generation(self.base_url)
        except Exception:
            pass

    def _start_two(self):
        # Submit the first request and wait for its first token before starting
        # the second: with max-running-requests=1, arrival order decides which
        # request owns the running slot.
        first = start_stream(self.base_url, SHORT_PROMPT, max_new_tokens=512)
        wait_for_first_token(first, label="first request")

        second = start_stream(self.base_url, SHORT_PROMPT, max_new_tokens=512)
        # Give the second request time to reach the scheduler; it can only be
        # queued while the first request is running.
        time.sleep(0.5)
        self.assertFalse(second.events, "second request was not queued")
        self.assertTrue(second.thread.is_alive(), "queued request finished early")
        return [first, second]

    def _assert_completed(self, requests_, expected_tokens=512):
        for idx, request in enumerate(requests_):
            self.assertFalse(request.errors, f"stream {idx} failed: {request.errors}")
            self.assertFalse(request.thread.is_alive(), f"stream {idx} did not end")
            output_ids = list(request.events[-1].get("output_ids", []))
            self.assertEqual(
                len(output_ids),
                expected_tokens,
                f"request {idx} did not complete: {len(output_ids)} tokens",
            )
            self.assertEqual(
                tuple(output_ids[: len(self.baseline[0])]),
                self.baseline[0],
                f"request {idx} changed the greedy rollout",
            )

    def test_pause_abort_drains_running_and_queued(self):
        requests_ = self._start_two()

        pause_generation(self.base_url, mode="abort")
        for request in requests_:
            request.thread.join(timeout=60)

        for idx, request in enumerate(requests_):
            self.assertFalse(request.errors, f"stream {idx} failed: {request.errors}")
            self.assertFalse(request.thread.is_alive(), f"stream {idx} did not end")
            self.assertTrue(request.events, f"stream {idx} produced no final event")
            last = request.events[-1]
            finish_type = last.get("meta_info", {}).get("finish_reason", {}).get("type")
            self.assertEqual(
                finish_type,
                "abort",
                f"request {idx} was not aborted by pause: {last}",
            )
            self.assertLess(
                len(last.get("output_ids", [])),
                512,
                f"request {idx} produced the full completion",
            )

        continue_generation(self.base_url)
        self.assertTrue(generate(self.base_url, SHORT_PROMPT, max_new_tokens=4))

    def test_pause_in_place_freezes_running_and_queued(self):
        requests_ = self._start_two()

        pause_generation(self.base_url, mode="in_place")
        time.sleep(0.5)
        frozen_tokens = len(requests_[0].events[-1].get("output_ids", []))
        self.assertGreater(frozen_tokens, 0)

        time.sleep(0.5)
        self.assertEqual(
            len(requests_[0].events[-1].get("output_ids", [])), frozen_tokens
        )
        self.assertFalse(requests_[1].events, "queued request started while paused")

        continue_generation(self.base_url)
        for request in requests_:
            request.thread.join(timeout=60)
        self._assert_completed(requests_)

    def test_pause_retract_preserves_queued_request(self):
        requests_ = self._start_two()

        pause_generation(self.base_url, mode="retract")
        time.sleep(0.2)
        continue_generation(self.base_url)
        for request in requests_:
            request.thread.join(timeout=60)
        self._assert_completed(requests_)


if __name__ == "__main__":
    unittest.main()
