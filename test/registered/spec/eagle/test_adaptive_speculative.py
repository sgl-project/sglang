import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=160, stage="base-b", runner_config="1-gpu-large")

HIGH_ACCEPT_PROMPT = (
    "Output exactly 128 new lines. "
    "Every line must be READY. "
    "Do not add numbering, punctuation, or commentary."
)

LOW_ACCEPT_PROMPT = (
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. "
    "Make it emotionally resonant and at least 100 words."
)

MAX_UPSHIFT_ATTEMPTS = 4
MAX_DOWNSHIFT_ATTEMPTS = 6


class TestAdaptiveSpeculativeServer(CustomTestCase):
    """Test adaptive speculative decoding with state switching and GSM8K accuracy."""

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {
                        "candidate_steps": [1, 3],
                        "ema_alpha": 1.0,
                        "warmup_batches": 1,
                        "update_interval": 1,
                        "up_hysteresis": 0.0,
                    },
                },
                f,
            )
            cls.adaptive_config_path = f.name

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "triton",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                    "--enable-metrics",
                    "--skip-server-warmup",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.adaptive_config_path):
            os.unlink(cls.adaptive_config_path)

    def _get_internal_state(self) -> dict:
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["internal_states"][0]

    def _scrape_metric(self, name: str, **label_filter) -> float | None:
        """Return the value of a Prometheus sample line, or None if absent.

        Matches a line whose metric name is exactly *name* (next char is '{'
        or whitespace) and whose labels include every key=value in
        *label_filter*.
        """
        text = requests.get(self.base_url + "/metrics", timeout=30).text
        for line in text.splitlines():
            if line.startswith("#") or not line.startswith(name):
                continue
            rest = line[len(name) :]
            if rest and rest[0] not in "{ ":
                continue
            if all(f'{k}="{v}"' in line for k, v in label_filter.items()):
                return float(line.rsplit(" ", 1)[1])
        return None

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _drive_upshift(self) -> dict:
        """Send high-acceptance prompts until steps upshift to 3."""
        state = self._get_internal_state()
        for _ in range(MAX_UPSHIFT_ATTEMPTS):
            self._generate(HIGH_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == 3:
                return state
        return state

    def _drive_downshift(self) -> dict:
        """Send low-acceptance prompts until steps downshift to 1."""
        state = self._get_internal_state()
        for _ in range(MAX_DOWNSHIFT_ATTEMPTS):
            self._generate(LOW_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == 1:
                return state
        return state

    def test_gsm8k_after_adaptive_switches(self):
        """Exercise up/down/up adaptive switches, then verify GSM8K accuracy."""
        state = self._drive_upshift()
        self.assertEqual(state["speculative_num_steps"], 3, f"Never upshifted: {state}")

        state = self._drive_downshift()
        self.assertEqual(
            state["speculative_num_steps"], 1, f"Never downshifted: {state}"
        )

        self._drive_upshift()

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"GSM8K after adaptive switches: {metrics}")
        self.assertGreater(metrics["score"], 0.20)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_accept_len = server_info["internal_states"][0]["avg_spec_accept_length"]
        print(f"avg_spec_accept_length={avg_accept_len:.4f}")

    def test_adaptive_metrics_exposed(self):
        """After an upshift, the adaptive current-state gauges are scrapeable."""
        state = self._drive_upshift()
        self.assertEqual(state["speculative_num_steps"], 3, f"Never upshifted: {state}")
        # One more decode so the reporter emits a fresh logging interval.
        self._generate(HIGH_ACCEPT_PROMPT)

        steps = self._scrape_metric("sglang:spec_num_steps")
        draft_tokens = self._scrape_metric("sglang:spec_num_draft_tokens")

        self.assertIn(steps, {1.0, 3.0}, "spec_num_steps gauge has unexpected value")
        self.assertIn(
            draft_tokens,
            {2.0, 4.0},
            "spec_num_draft_tokens gauge has unexpected value",
        )


class TestAdaptiveZeroStepBatchSizeServer(CustomTestCase):
    """steps=0 (nospec) fallback triggered by batch size.

    Config routes BS>=8 -> steps=0 (drafting disabled) and BS<8 -> steps=3.
    Verifies (1) a concurrent burst drives the worker to steps=0, and (2) a
    sequence decoded at steps=0 recovers full draft acceptance once it returns
    to steps=3 -- i.e. draft_extend keeps the draft KV synced while drafting is
    off. If it didn't, post-recovery drafts would be rejected (~0 accepts).
    """

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    base_url = DEFAULT_URL_FOR_TEST

    COUNT_PROMPT = "Count from 1 to 400, separated by commas. Output only the numbers."

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {"candidate_steps": [3], "warmup_batches": 0},
                    "8": {"candidate_steps": [0], "warmup_batches": 0},
                },
                f,
            )
            cls.adaptive_config_path = f.name

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "triton",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                    "--max-running-requests",
                    "32",
                    "--skip-server-warmup",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.adaptive_config_path):
            os.unlink(cls.adaptive_config_path)

    def _steps(self) -> int:
        r = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["internal_states"][0]["speculative_num_steps"]

    def _generate(self, max_new_tokens: int, hold: dict | None = None) -> dict:
        r = requests.post(
            self.base_url + "/generate",
            json={
                "text": self.COUNT_PROMPT,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        if hold is not None:
            hold["meta"] = out["meta_info"]
            hold["text"] = out["text"]
        return out["meta_info"]

    def _wait_until_steps(self, target: int, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self._steps() == target:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def test_batch_size_triggers_zero_step(self):
        """A concurrent burst (BS>=8) routes the worker to steps=0; draining
        back to BS<8 routes it back to steps=3."""
        self.assertEqual(self._steps(), 3, "expected initial steps=3")

        burst = [
            threading.Thread(target=self._generate, args=(400,)) for _ in range(14)
        ]
        for t in burst:
            t.start()
        reached_zero = self._wait_until_steps(0, timeout=30.0)
        for t in burst:
            t.join()
        self.assertTrue(reached_zero, "batch size did not drive steps to 0")

        # Drain complete: a single small request runs at BS=1 -> steps=3.
        self._generate(16)
        self.assertEqual(self._steps(), 3, "did not route back to steps=3 at BS=1")

    def test_zero_step_within_sequence_recovery(self):
        """A sequence decoded at steps=0 (during a burst) recovers full draft
        acceptance after the burst drains and it returns to steps=3."""
        burst = [
            threading.Thread(target=self._generate, args=(400,)) for _ in range(14)
        ]
        for t in burst:
            t.start()
        self.assertTrue(
            self._wait_until_steps(0, timeout=30.0), "burst did not reach steps=0"
        )

        # Submit a longer probe INTO the steps=0 batch; it outlives the burst,
        # so its tail decodes at steps=3 after the burst drains.
        hold: dict = {}
        probe = threading.Thread(target=self._generate, args=(700, hold))
        probe.start()
        for t in burst:
            t.join()
        probe.join()

        hist = hold["meta"]["spec_correct_drafts_histogram"]
        self.assertGreater(
            hist[0], 0, f"probe was never decoded at steps=0 (hist={hist})"
        )
        recovered = sum(hist[1:])
        self.assertGreater(
            recovered,
            5,
            f"probe saw ~0 draft acceptance after returning to steps=3 -> draft KV "
            f"likely stale during steps=0 (hist={hist})",
        )
        self.assertGreater(
            len(hold["text"].strip()), 0, "empty output across the excursion"
        )


if __name__ == "__main__":
    unittest.main()
