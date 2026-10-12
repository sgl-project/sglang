"""MI35x DeepSeek-V4.1-Flash GSM8K Completion Evaluation Test (4-GPU)

Tests deepseek-ai/DeepSeek-V4.1-Flash with the GSM8K few-shot benchmark on
MI35x.

Server arguments follow the MI350X High-Throughput cell of the cookbook recipe
(docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx): TP4 + EP4 with
DSpark speculative decoding, decode CUDA graphs up to the 256-request cap, and
decoder SWA bounded replay. The attention and MoE backends are left to resolve
on HIP, as the recipe does. SGLANG_USE_AITER and SGLANG_MOE_PADDING come from
the ROCm image (docker/rocm.Dockerfile), so neither the recipe nor this test
sets them.

DeepSeek-V4.1-Flash ships FP4 experts, which need gfx95x, so this runs on MI35x
only. The recipe itself is TP4, so the job takes an 8-GPU MI35x runner and uses
four of its GPUs, like the MiniMax TP4 jobs.

The eval uses the few-shot *completion* harness, as the DeepSeek-V4-Flash MI35x
test does. It bypasses the chat template, so V4.1's thinking mode cannot route
the answer into `reasoning_content` and leave `message.content` empty. Scoring
raw completions keeps this test measuring whether the ROCm kernels produce
correct tokens.

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

# Register for AMD CI - DeepSeek-V4.1-Flash accuracy test on MI35x (~18 min in
# run 36797490320: ~16 min of server startup, then 1319 GSM8K questions in ~1 min)
register_amd_ci(
    est_time=1200, suite="nightly-amd-4-gpu-mi35x-deepseek-v41-flash", nightly=True
)

DEEPSEEK_V41_FLASH_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V41_FLASH_MODEL_PATH", "deepseek-ai/DeepSeek-V4.1-Flash"
)
SERVER_LAUNCH_TIMEOUT = 5400
# Measured 0.911 on all 1319 questions with the High-Throughput cell at TP4 on
# MI35x (rocm10, sgl-project/sglang Actions run 36797490320). The threshold
# sits 0.02 under that.
ACCURACY_THRESHOLD = 0.89
TP_SIZE = 4


class TestDeepseekV41FlashEvalMI35x(CustomTestCase):
    """DeepSeek-V4.1-Flash GSM8K Completion Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
        cls.max_new_tokens = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "512"))

    def test_deepseek_v41_flash_gsm8k_accuracy(self):
        """Test DeepSeek-V4.1-Flash with GSM8K few-shot completion benchmark."""
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--ep-size",
            str(TP_SIZE),
            "--mem-fraction-static",
            "0.8",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--cuda-graph-max-bs-decode",
            "256",
            "--max-running-requests",
            "256",
            "--enable-decoder-swa-bounded-replay",
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
        # Both come from the recipe. AITER_FLYDSL_FORCE_REDUCE forces the FlyDSL
        # MoE down-projection onto a per-slot reduce instead of atomics, which
        # makes the output repeatable run to run.
        env["AITER_FLYDSL_FORCE_REDUCE"] = "1"
        # At the default bound, small batches take AITER's BF16-activation MoE
        # route, which has no kernel for these experts at the current pin: the
        # server fails during decode-graph warmup. Keep this until
        # ROCm/aiter#5802 is in the AITER pin (see sgl-project/sglang#41308).
        # AITER reads it directly, so it will not grep to anything in-tree.
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
                summary = "### DeepSeek-V4.1-Flash Model (MI35x)\n\n"
                summary += "| Model | TP | Accuracy | Threshold | Status |\n"
                summary += "| ----- | -- | -------- | --------- | ------ |\n"
                summary += f"| {DEEPSEEK_V41_FLASH_MODEL_PATH} | {TP_SIZE} | {acc:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                acc,
                ACCURACY_THRESHOLD,
                f"DeepSeek-V4.1-Flash accuracy {acc:.3f} below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
