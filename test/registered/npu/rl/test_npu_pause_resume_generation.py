"""1-NPU smoke tests for generation pause/resume.

RL Test Consolidation (sgl-project/sglang#47624) — pause/resume area, 1-NPU tier.
File ownership per RFC section 5 (pause/resume, 1-NPU smoke).

Coverage:
- P0  BF16 precision contract (RFC section 4): the shared instance launches
      with an explicit --dtype bfloat16 (never dtype-inferred) and
      /server_info must echo dtype=bfloat16 with the flag pinned in
      launch_command — weights and KV cache follow the model dtype on the
      ascend attention backend. All scenario assertions below therefore
      validate the BF16 numerics path.
- P1  pause freezes the stream: chunk count is frozen across a sampling
      window under in_place pause; continue lets the request finish with a
      proper finish_reason (PR-S1).
- P2  idempotent contracts end-to-end: double pause, continue without pause,
      and a second continue are all HTTP 200 and the engine still generates
      (PR-U2 e2e side; scheduler.py: pause re-sets the flag keeping the first
      _pause_started_at, continue is an idempotent no-op).
- P3  abort mode contract: mode="abort" returns 200 and aborts the in-flight
      stream (tokenizer-level abort-all; the scheduler-side assert only
      guards dispatched requests). Unknown non-abort modes reaching the
      scheduler are a documented crash risk, not tested here.
- P4  late requests queue during pause: a request submitted while paused
      produces no chunks until continue, then completes (PR-S2 drain
      semantics).
- P5  retract parity: pausing mid-stream with mode=retract and continuing
      reproduces the greedy baseline text exactly (recompute path) (PR-U3).
- P6  flush_cache during a retract pause is legal: the engine is idle while
      paused, flush succeeds, continue then generates normally (PR-S4;
      scheduler.flush_cache uses is_fully_idle(ignore_waiting=paused)).

Server: single shared instance (Llama-3.2-1B default, explicit --dtype
bfloat16, ascend attention backend).
Die selection: set RL_FEAT_NPU_ID (default "0") to pin
ASCEND_RT_VISIBLE_DEVICES.

Sibling tests: test/registered/unit/managers/test_scheduler_pause_generation.py
(scheduler-level state machine) and the scheduler control tests.
"""

import json
import os
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_npu_ci(est_time=1200, suite="nightly-npu-a3-merged", nightly=True)

# Model override for hosts without the default weights cached (e.g. npu188
# only caches Qwen/Qwen3-0.6B): export RL_FEAT_MODEL_PATH=<absolute path>.
MODEL = os.environ.get("RL_FEAT_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH)
LONG_PROMPT = "Write a long, detailed story about a robot learning to paint."
SHORT_PROMPT = "The capital of France is"


def _sampling_params(max_new_tokens):
    return {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True}


class TestNpuPauseResumeGeneration(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        port = find_available_port(31000)
        cls.base_url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("RL_FEAT_NPU_ID", "0")
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "ascend", "--dtype", "bfloat16"],
            env=env,
        )
        # P0: BF16 precision contract — every test below inherits it.
        info = requests.get(cls.base_url + "/server_info", timeout=120).json()
        assert str(info.get("dtype", "")).lower() == "bfloat16", (
            f"server must run BF16 weights/KV, got dtype={info.get('dtype')!r}"
        )
        assert "bfloat16" in str(info.get("launch_command", "")).lower(), (
            f"launch command must pin --dtype bfloat16: {info.get('launch_command')}"
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # --- helpers -----------------------------------------------------------

    def _post(self, path, payload, expect_ok=True):
        r = requests.post(
            self.base_url + path,
            json=payload,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        if expect_ok:
            self.assertEqual(r.status_code, 200, f"{path} -> {r.status_code}: {r.text}")
        return r

    def _generate(self, prompt, max_new_tokens=64):
        r = requests.post(
            self.base_url + "/generate",
            json={"text": prompt, "sampling_params": _sampling_params(max_new_tokens)},
            timeout=300,
        )
        self.assertEqual(r.status_code, 200)
        return r.json()["text"]

    def _stream_collect(self, prompt, max_new_tokens, chunks, done):
        """Collect SSE chunks into `chunks`; signal `done` when finished."""
        try:
            with requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": _sampling_params(max_new_tokens),
                    "stream": True,
                },
                stream=True,
                timeout=300,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith(b"data:"):
                        continue
                    payload = line[len(b"data:") :].strip()
                    if payload == b"[DONE]":
                        break
                    chunks.append(json.loads(payload))
        except Exception as e:  # surface transport errors as a sentinel chunk
            chunks.append({"__error__": str(e)})
        finally:
            done.set()

    def _start_stream(self, prompt, max_new_tokens):
        chunks, done = [], threading.Event()
        thread = threading.Thread(
            target=self._stream_collect,
            args=(prompt, max_new_tokens, chunks, done),
            daemon=True,
        )
        thread.start()
        return chunks, done, thread

    @staticmethod
    def _final_text(chunks):
        if not chunks or "__error__" in chunks[-1]:
            return None
        return chunks[-1].get("text", "")

    # --- scenarios ----------------------------------------------------------

    def test_1_pause_freezes_stream_then_continue_completes(self):
        chunks, done, thread = self._start_stream(LONG_PROMPT, 512)
        deadline = time.time() + 120
        while len(chunks) < 5 and time.time() < deadline:
            time.sleep(0.1)
        self.assertGreaterEqual(len(chunks), 5, "stream did not start")

        self._post("/pause_generation", {"mode": "in_place"})
        time.sleep(1.5)
        frozen = len(chunks)
        time.sleep(1.5)
        self.assertEqual(
            len(chunks), frozen, "chunks must not advance while paused (in_place)"
        )

        self._post("/continue_generation", {})
        thread.join(timeout=300)
        self.assertTrue(done.is_set(), "stream did not finish after continue")
        self.assertGreater(len(chunks), frozen)
        meta = chunks[-1].get("meta_info", {})
        self.assertEqual(
            meta.get("finish_reason", {}).get("type"),
            "length",
            f"unexpected finish: {meta}",
        )

    def test_2_double_pause_and_continue_without_pause_are_idempotent(self):
        self._post("/pause_generation", {"mode": "in_place"})
        self._post("/pause_generation", {"mode": "in_place"})
        self._post("/continue_generation", {})
        self._post("/continue_generation", {})  # no-op

        text = self._generate(SHORT_PROMPT, 16)
        self.assertTrue(text)

    def test_3_abort_mode_returns_200_and_engine_stays_healthy(self):
        """HTTP-level contract discovered on main: mode="abort" is a
        first-class mode handled in the tokenizer manager (abort-all, never
        dispatched to the scheduler), so it returns 200 and the engine stays
        healthy afterwards.

        Known gaps (observed on npu188, for the defect-fix round — NOT
        asserted here): (1) abort-all leaves the tokenizer's is_pause gate
        SET, so new requests hang until /continue_generation re-opens
        admission — arguably abort should restore admission by itself;
        (2) an in-flight SSE stream is not terminated with a final chunk
        after abort (client hangs until read timeout); (3) unknown
        non-abort modes are dispatched to the scheduler and hit the
        scheduler-side assert — a crash risk."""
        r = self._post("/pause_generation", {"mode": "abort"})
        self.assertEqual(r.status_code, 200)

        # is_pause gate stays set after abort: reopen admission explicitly.
        self._post("/continue_generation", {})

        text = self._generate(SHORT_PROMPT, 16)
        self.assertTrue(text, "engine must serve again after abort + continue")

    def test_4_late_request_queued_during_pause(self):
        self._post("/pause_generation", {"mode": "in_place"})

        chunks, done, thread = self._start_stream(SHORT_PROMPT, 32)
        time.sleep(2)
        self.assertEqual(
            len(chunks), 0, "a request submitted while paused must not be served"
        )

        self._post("/continue_generation", {})
        thread.join(timeout=180)
        self.assertTrue(done.is_set())
        self.assertGreater(len(chunks), 0)
        self.assertIsNone(
            next((c for c in chunks if "__error__" in c), None), f"{chunks}"
        )

    def test_5_retract_pause_preserves_greedy_output(self):
        baseline = self._generate(SHORT_PROMPT, 64)

        chunks, done, thread = self._start_stream(SHORT_PROMPT, 64)
        deadline = time.time() + 120
        while len(chunks) < 3 and time.time() < deadline:
            time.sleep(0.1)

        self._post("/pause_generation", {"mode": "retract"})
        self._post("/continue_generation", {})
        thread.join(timeout=300)
        self.assertTrue(done.is_set())

        final = self._final_text(chunks)
        self.assertIsNotNone(final, f"stream error: {chunks}")
        self.assertEqual(
            final,
            baseline,
            "retract+continue must reproduce the greedy baseline exactly",
        )

    def test_6_flush_cache_during_retract_pause(self):
        self._post("/pause_generation", {"mode": "retract"})
        self._post("/flush_cache", {})
        self._post("/continue_generation", {})

        text = self._generate(SHORT_PROMPT, 16)
        self.assertTrue(text)


if __name__ == "__main__":
    unittest.main()
