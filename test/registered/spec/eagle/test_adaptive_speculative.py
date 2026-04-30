import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
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

register_cuda_ci(est_time=76, suite="stage-b-test-1-gpu-large")

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
                    "candidate_steps": [1, 3],
                    "ema_alpha": 1.0,
                    "warmup_batches": 1,
                    "update_interval": 1,
                    "up_hysteresis": 0.0,
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


class TestAdaptiveZeroStepServerV2(CustomTestCase):
    """Steps=0 fallback under spec v2: drafting is disabled but draft_extend
    still runs so the draft model's KV cache stays in sync, letting the
    controller re-enable spec decoding via a probe upshift."""

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "candidate_steps": [0, 3],
                    "ema_alpha": 1.0,
                    "warmup_batches": 0,
                    "update_interval": 1,
                    # Drop threshold to steps=0 = prev - 0.5 + down_hysteresis = 0.2.
                    "down_hysteresis": 0.7,
                    # Prevent immediate bounce-back out of steps=0: rise threshold
                    # = 0 - 0.5 + 0.51 = 0.01, and reported accept is always 0
                    # while drafting is disabled.
                    "up_hysteresis": 0.51,
                    # Force periodic upshift so the controller can re-observe
                    # acceptance (otherwise steps=0 is absorbing).
                    "zero_step_probe_interval": 5,
                },
                f,
            )
            cls.adaptive_config_path = f.name

        try:
            with envs.SGLANG_ENABLE_SPEC_V2.override(True):
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

    def _state(self) -> int:
        r = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["internal_states"][0]["speculative_num_steps"]

    def _generate(self, prompt: str, max_new_tokens: int = 32) -> dict:
        r = requests.post(
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
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_downshift_to_zero_and_generation(self):
        """Low-accept prompts push the controller into steps=0; generation at
        steps=0 must still produce coherent, non-empty output."""
        self.assertEqual(self._state(), 3, "expected initial steps=3")

        for _ in range(3):
            self._generate(LOW_ACCEPT_PROMPT, max_new_tokens=32)

        self.assertEqual(
            self._state(),
            0,
            "controller did not downshift to steps=0 after low-accept prompts",
        )

        output = self._generate("The capital of France is", max_new_tokens=16)
        text = output["text"].strip()
        self.assertGreater(len(text), 0, "empty output at steps=0")
        self.assertIn("Paris", text, f"incoherent output at steps=0: {text!r}")

    def test_probe_upshift_warms_draft_kv(self):
        """After entering steps=0, a probe upshift must see non-trivial draft
        acceptance — if draft_extend during steps=0 didn't keep the draft KV
        cache in sync, the upshifted batches would show 0 acceptance."""
        for _ in range(3):
            self._generate(LOW_ACCEPT_PROMPT, max_new_tokens=32)
        self.assertEqual(self._state(), 0, "precondition: should be at steps=0")

        # Enough tokens to trigger at least one probe upshift
        # (zero_step_probe_interval=5).
        output = self._generate("The capital of France is", max_new_tokens=48)
        hist = output["meta_info"]["spec_accept_histogram"]
        self.assertGreater(
            len(hist),
            1,
            f"probe upshift did not execute any drafting; hist={hist}",
        )
        self.assertGreater(
            sum(hist[1:]),
            0,
            f"upshifted batches saw 0 draft acceptance → draft KV likely stale "
            f"(hist={hist})",
        )


if __name__ == "__main__":
    unittest.main()
