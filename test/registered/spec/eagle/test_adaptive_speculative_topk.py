import json
import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-large")

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

# candidate_steps=[2, 3] is required at topk=4: min_step=1 would only yield a
# pool of 4 < num_draft_tokens-1=15 and fail _validate_candidate_steps_against_topk.
EXPECTED_UPSHIFT_STEPS = 3
EXPECTED_DOWNSHIFT_STEPS = 2


class TestAdaptiveSpeculativeTopkServer(CustomTestCase):
    """Adaptive speculative decoding under tree EAGLE (topk > 1).

    Exercises tier swap correctness: both runtime states init under topk=4,
    upshift drives steps to 3, downshift drives to 2 (the floor of
    candidate_steps=[2, 3], not 1), and a smoke decode after the second
    upshift confirms the verify path runs cleanly post-swap.
    """

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "candidate_steps": [2, 3],
                    "ema_alpha": 0.3,
                    "warmup_batches": 1,
                    "update_interval": 1,
                    "up_hysteresis": 0.0,
                    "down_hysteresis": -0.5,
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
                    "flashinfer",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "4",
                    "--speculative-num-draft-tokens",
                    "16",
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

    def _drive_upshift(self, target: int = EXPECTED_UPSHIFT_STEPS) -> dict:
        state = self._get_internal_state()
        for _ in range(MAX_UPSHIFT_ATTEMPTS):
            self._generate(HIGH_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == target:
                return state
        return state

    def _drive_downshift(self, target: int = EXPECTED_DOWNSHIFT_STEPS) -> dict:
        state = self._get_internal_state()
        for _ in range(MAX_DOWNSHIFT_ATTEMPTS):
            self._generate(LOW_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == target:
                return state
        return state

    def test_adaptive_switches_under_topk_gt_1(self):
        """Up -> down -> up sequence under topk=4, then a smoke decode."""
        state = self._drive_upshift()
        self.assertEqual(
            state["speculative_num_steps"],
            EXPECTED_UPSHIFT_STEPS,
            f"Never upshifted: {state}",
        )

        state = self._drive_downshift()
        self.assertEqual(
            state["speculative_num_steps"],
            EXPECTED_DOWNSHIFT_STEPS,
            f"Never downshifted: {state}",
        )

        state = self._drive_upshift()
        self.assertEqual(
            state["speculative_num_steps"],
            EXPECTED_UPSHIFT_STEPS,
            f"Never re-upshifted: {state}",
        )

        # Smoke decode: confirms the verify path runs cleanly after the
        # last tier swap; a non-empty output is enough signal here, broader
        # parity is left to local-only smokes.
        result = self._generate("Hello", max_new_tokens=8)
        self.assertTrue(result.get("text"), f"Empty decode after swap: {result}")


if __name__ == "__main__":
    unittest.main()
