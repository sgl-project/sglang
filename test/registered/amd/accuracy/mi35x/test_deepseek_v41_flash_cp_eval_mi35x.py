"""MI35x DeepSeek-V4.1-Flash GSM8K Completion Evaluation Test with prefill CP (4-GPU)

Tests deepseek-ai/DeepSeek-V4.1-Flash with the GSM8K few-shot benchmark on MI35x
under interleave prefill context parallelism: TP4 with CP4, DSpark speculative
decoding and decoder SWA bounded replay, so its prefills run the HIP CP
low-ratio sources, the CP layout of the late-layer tail and the scatter of the
DSpark captures back into each rank's shard. The vision tower does not run under
CP, so the server is text only.

The eval uses the few-shot *completion* harness, as the non-CP DeepSeek-V4.1-Flash
MI35x test does, so it measures whether the ROCm kernels produce correct tokens.

Registry: nightly-amd-4-gpu-mi35x-deepseek-v41-flash suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=1200, suite="nightly-amd-4-gpu-mi35x-deepseek-v41-flash", nightly=True
)

DEEPSEEK_V41_FLASH_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V41_FLASH_MODEL_PATH", "deepseek-ai/DeepSeek-V4.1-Flash"
)
SERVER_LAUNCH_TIMEOUT = 5400
# Measured 0.915 on all 1319 questions at TP4 + CP4 on MI35x (rocm10, image
# rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261102). The threshold is the non-CP test's.
ACCURACY_THRESHOLD = 0.89
TP_SIZE = 4


class TestDeepseekV41FlashCPEvalMI35x(CustomTestCase):
    """DeepSeek-V4.1-Flash GSM8K Completion Evaluation Test with prefill CP for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
        cls.max_new_tokens = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "512"))

    def test_deepseek_v41_flash_cp_gsm8k_accuracy(self):
        """Test DeepSeek-V4.1-Flash under prefill CP with GSM8K few-shot completion benchmark."""
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-decoder-swa-bounded-replay",
            "--language-model-only",
            "--mem-fraction-static",
            "0.8",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--enforce-shared-experts-fusion",
            "--fp8-gemm-backend",
            "aiter",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs-decode",
            "128",
            "--max-running-requests",
            "128",
            "--reasoning-parser",
            "auto",
            "--tool-call-parser",
            "auto",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]
        env = os.environ.copy()
        # Both as in the non-CP DeepSeek-V4.1-Flash MI35x test: a repeatable FlyDSL MoE
        # down-projection, and no BF16-activation MoE route (ROCm/aiter#5802).
        env["AITER_FLYDSL_FORCE_REDUCE"] = "1"
        env["AITER_BF16_FP8_MOE_BOUND"] = "0"

        process = popen_launch_server(
            DEEPSEEK_V41_FLASH_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

        try:
            requests.get(self.base_url + "/flush_cache")

            args = SimpleNamespace(
                num_shots=8,
                data_path=None,
                num_questions=self.num_questions,
                parallel=self.num_questions,
                max_new_tokens=self.max_new_tokens,
                host="127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            acc = metrics["accuracy"]

            passed = acc >= ACCURACY_THRESHOLD
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  accuracy={acc:.3f} threshold={ACCURACY_THRESHOLD} {status}")

            if is_in_ci():
                summary = "### DeepSeek-V4.1-Flash Model, prefill CP (MI35x)\n\n"
                summary += "| Model | TP | CP | Accuracy | Threshold | Status |\n"
                summary += "| ----- | -- | -- | -------- | --------- | ------ |\n"
                summary += f"| {DEEPSEEK_V41_FLASH_MODEL_PATH} | {TP_SIZE} | {TP_SIZE} | {acc:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                acc,
                ACCURACY_THRESHOLD,
                f"DeepSeek-V4.1-Flash prefill CP accuracy {acc:.3f} below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
