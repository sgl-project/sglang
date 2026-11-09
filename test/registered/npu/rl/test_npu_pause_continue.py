"""P0 NPU tests: pause/continue generation around RL weight updates.

The scheduler must drain or freeze in-flight requests without emitting tokens
from a half-updated model, and must resume the same requests afterwards.
"""

import os
import threading
import time
import unittest

import requests

from sglang.test.ascend.npu_rl_utils import (
    begin_weight_update,
    checkpoint_tensor_index,
    continue_generation,
    end_weight_update,
    generate,
    generation_signature,
    load_checkpoint_tensor,
    npu_available,
    npu_server,
    pause_generation,
    stream_generate,
    update_weights_from_tensor,
    warmup_npu,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

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
    "--mem-fraction-static",
    "0.55",
    "--max-model-len",
    "1024",
    "--max-running-requests",
    "8",
)
SHORT_PROMPT = PROMPTS[0]
NORM_WEIGHT = "model.norm.weight"


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuPauseContinue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        warmup_npu()
        cls._server = npu_server(MODEL, extra_args=SERVER_ARGS)
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

    def _stream_async(self, events, errors):
        def _run():
            try:
                for event in stream_generate(
                    self.base_url, SHORT_PROMPT, max_new_tokens=128
                ):
                    events.append(event)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread

    def test_pause_abort_returns_inflight(self):
        results = {}
        thread = self._decode_async(512, results)

        time.sleep(0.5)
        pause_generation(self.base_url, mode="abort")
        thread.join(timeout=60)

        self.assertNotIn("error", results)
        response = results.get("response", {})
        output_ids = response.get("output_ids", [])
        finish_type = response.get("meta_info", {}).get("finish_reason", {}).get("type")
        self.assertTrue(
            finish_type == "abort" or len(output_ids) < 512,
            "pause(abort) returned before the in-flight request was aborted",
        )

        continue_generation(self.base_url)
        self.assertTrue(generate(self.base_url, SHORT_PROMPT, max_new_tokens=4))

    def test_pause_retract_recomputes_kv(self):
        results = {}
        thread = self._decode_async(64, results)

        time.sleep(0.2)
        pause_generation(self.base_url, mode="retract")
        time.sleep(0.2)
        continue_generation(self.base_url)
        thread.join(timeout=60)

        self.assertNotIn("error", results)
        self.assertEqual(
            tuple(results["response"]["output_ids"][: len(self.baseline[0])]),
            self.baseline[0],
            "retract/continue changed a greedy rollout",
        )

    def test_pause_in_place_freezes_output(self):
        events, errors = [], []
        thread = self._stream_async(events, errors)

        time.sleep(0.5)
        pause_generation(self.base_url, mode="in_place")
        time.sleep(0.5)
        frozen_at = len(events)
        time.sleep(1.0)
        self.assertEqual(
            len(events),
            frozen_at,
            "in-flight request kept emitting tokens while generation was paused",
        )

        continue_generation(self.base_url)
        thread.join(timeout=60)

        self.assertFalse(errors, f"stream failed: {errors}")
        self.assertGreater(len(events), frozen_at, "stream did not resume")

    def test_pause_then_update_then_continue(self):
        original = load_checkpoint_tensor(self.tensor_index, NORM_WEIGHT)
        updated = original.new_full(original.shape, 0.5)

        pause_generation(self.base_url, mode="in_place")
        begin_weight_update(self.base_url)
        update_weights_from_tensor(
            self.base_url,
            [(NORM_WEIGHT, updated.to("npu"))],
            weight_version="paused-v1",
        )
        end_weight_update(self.base_url)
        continue_generation(self.base_url)

        response = generate(self.base_url, SHORT_PROMPT, max_new_tokens=8)
        self.assertEqual(
            response["meta_info"]["weight_version"],
            "paused-v1",
            "resumed request was not served by the updated weights",
        )

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
        self.assertNotEqual(second.status_code, 500, "second pause crashed the server")
        continue_generation(self.base_url)
        self.assertTrue(generate(self.base_url, SHORT_PROMPT, max_new_tokens=4))


if __name__ == "__main__":
    unittest.main()
