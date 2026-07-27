"""Test Adaptive Speculative Decoding on NPU.

[Test Category] Speculative Decoding
[Test Target] --speculative-algorithm=EAGLE3; --speculative-adaptive;
--speculative-adaptive-config; --speculative-num-steps (dynamic);
--speculative-eagle-topk; --speculative-num-draft-tokens
[Platform] NPU (Ascend A3, CANN 9.0.0)
[Porting Source] Ported from GPU: sgl-project/sglang test/test_adaptive_speculative.py
  Class: TestAdaptiveSpeculativeServer
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


NPU_ENV = {
    **os.environ,
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "0",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}

# High-acceptance prompt: easy to predict, high draft token acceptance
HIGH_ACCEPT_PROMPT = (
    "Output exactly 128 new lines. "
    "Every line must be READY. "
    "Do not add numbering, punctuation, or commentary."
)

# Low-acceptance prompt: creative, low draft token acceptance
LOW_ACCEPT_PROMPT = (
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. "
    "Make it emotionally resonant and at least 100 words."
)

MAX_UPSHIFT_ATTEMPTS = 4
MAX_DOWNSHIFT_ATTEMPTS = 6


class TestNPUAdaptiveSpeculativeServer(CustomTestCase):
    """Verify adaptive EAGLE3 upshift/downshift and GSM8K accuracy on NPU."""

    model = QWEN3_8B_WEIGHTS_PATH
    draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        # Create temp config.json with candidate_steps=[1, 3]
        # This constrains adaptive to only switch between 1 and 3,
        # making the upshift/downshift assertions deterministic.
        #
        # NPU B090 image requires batch-size keyed format:
        #   {"<bs>": {"candidate_steps": [...], ...}}
        # See error: "must contain at least one integer-string BS key,
        #   e.g. {"1": {"candidate_steps": [1,3,7]}}"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {
                        "candidate_steps": [1, 3],
                        "ema_alpha": 1.0,
                        "warmup_batches": 1,
                        "update_interval": 1,
                        "up_hysteresis": 0.0,
                    }
                },
                f,
            )
            cls.adaptive_config_path = f.name

        logger.info("Created adaptive config at: %s", cls.adaptive_config_path)
        logger.info("Model: %s", cls.model)
        logger.info("Draft model: %s", cls.draft_model)

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    "0.7",
                    "--tp-size",
                    "1",
                    "--sampling-backend",
                    "ascend",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "8",
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                ],
                env=NPU_ENV,
            )
            logger.info("Adaptive server started successfully.")
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
        """Get internal state from /server_info.

        Same as GPU version: internal_states[0] contains the adaptive state
        including speculative_num_steps and avg_spec_accept_length.
        """
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
        """Drive up/down/up adaptive switches, then verify GSM8K score > 0.69."""
        logger.info("=== Driving upshift (high-acceptance prompts) ===")
        state = self._drive_upshift()
        self.assertEqual(state["speculative_num_steps"], 3, f"Never upshifted: {state}")
        logger.info("Upshifted to num_steps=3: %s", state)

        logger.info("=== Driving downshift (low-acceptance prompts) ===")
        state = self._drive_downshift()
        self.assertEqual(
            state["speculative_num_steps"], 1, f"Never downshifted: {state}"
        )
        logger.info("Downshifted to num_steps=1: %s", state)

        logger.info("=== Driving upshift again ===")
        self._drive_upshift()

        logger.info("=== Running GSM8K ===")
        requests.get(self.base_url + "/flush_cache", timeout=30)

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="chat",
            max_tokens=2048,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        logger.info("GSM8K metrics (adaptive speculative): %s", metrics)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Adaptive Speculative on NPU)\n"
                f'{metrics["score"]=:.3f}\n'
            )

        # Qwen3-8B official GSM8K (thinking mode) ~0.92; EAGLE3 is lossless
        # speculation so score should match target. Threshold 0.80 leaves
        # ~13% margin for NPU precision variance and 200-example sampling.
        self.assertGreater(
            metrics["score"],
            0.80,
            "GSM8K score should be > 0.80 with adaptive speculative",
        )

        # Verify avg_spec_accept_length is reported (like GPU version)
        server_info = requests.get(self.base_url + "/server_info").json()
        avg_accept_len = server_info["internal_states"][0]["avg_spec_accept_length"]
        logger.info("avg_spec_accept_length=%.4f", avg_accept_len)


class TestNPUAdaptiveZeroStepBatchSize(CustomTestCase):
    """Verify adaptive steps=0 (nospec) fallback triggered by batch size on NPU.

    Config routes BS>=8 -> steps=0 (drafting disabled) and BS<8 -> steps=3, so
    the server cycles steps=3 -> steps=0 -> steps=3 as load rises and falls.
    Ported from GPU: TestAdaptiveZeroStepBatchSizeServer.
    """

    model = QWEN3_8B_WEIGHTS_PATH
    draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Reuse HIGH_ACCEPT_PROMPT (repetitive output) for high draft acceptance.
    # EAGLE3 on NPU has lower accept rate than EAGLE on GPU, so the original
    # COUNT_PROMPT only achieves ~0.22. HIGH_ACCEPT_PROMPT achieves >0.5.
    BS_PHASE_PROMPT = HIGH_ACCEPT_PROMPT

    @classmethod
    def setUpClass(cls):
        # NPU B090 image requires batch-size keyed format.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {"candidate_steps": [3], "warmup_batches": 0},
                    "8": {"candidate_steps": [0], "warmup_batches": 0},
                },
                f,
            )
            cls.adaptive_config_path = f.name

        logger.info("BS-phase config: %s", cls.adaptive_config_path)

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    "0.7",
                    "--tp-size",
                    "1",
                    "--sampling-backend",
                    "ascend",
                    "--speculative-algorithm",
                    "EAGLE3",
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
                ],
                env=NPU_ENV,
            )
            logger.info("BS-phase adaptive server started successfully.")
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

    def test_batch_size_step_cycle(self):
        """Server cycles steps=3 -> steps=0 -> steps=3 as load rises and falls."""
        one = {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}

        def generate_single() -> dict:
            r = requests.post(
                self.base_url + "/generate",
                json={"text": self.BS_PHASE_PROMPT, "sampling_params": one},
                timeout=600,
            )
            self.assertEqual(r.status_code, 200, r.text)
            return r.json()["meta_info"]

        # Phase 1: BS=1 -> steps=3, drafting active.
        # EAGLE3 on NPU achieves ~0.28 accept rate (vs EAGLE on GPU >0.8).
        # Threshold 0.2 confirms drafting is active without being too strict.
        m1 = generate_single()
        self.assertEqual(self._steps(), 3, "expected steps=3 at BS=1")
        self.assertGreater(
            m1["spec_accept_rate"], 0.2, f"not drafting at steps=3: {m1}"
        )

        # Phase 2: BS=14 -> steps=0 (BS>=8 disables drafting).
        full = {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True}
        r = requests.post(
            self.base_url + "/generate",
            json={"text": [self.BS_PHASE_PROMPT] * 14, "sampling_params": [full] * 14},
            timeout=600,
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self._steps(), 0, "BS>=8 did not switch to steps=0")

        # Phase 3: BS=1 -> steps=3 again, drafting restored.
        m3 = generate_single()
        self.assertEqual(self._steps(), 3, "did not reopen to steps=3")
        self.assertGreater(
            m3["spec_accept_rate"], 0.2, f"drafting not restored after steps=0: {m3}"
        )


if __name__ == "__main__":
    unittest.main()
