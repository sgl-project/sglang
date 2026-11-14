"""MI35x DeepSeek-V4.1-Flash GSM8K Evaluation Test with DSpark + FLy (4-GPU)

FLy (--speculative-use-fly, entropy-gated deferred verification) is a lossy DSpark
accept that only runs on ROCm. This test launches the setup of
test_deepseek_v41_flash_eval_mi35x.py (TP4 + EP4, DSpark block size 5) with FLy at
window 2 and entropy threshold 0.3, the setting vLLM evaluated FLy with on DSpark,
checks that the server reports FLy on, and that GSM8K accuracy holds.

The eval is sgl-eval's gsm8k (zero-shot chat, \\boxed{} extraction, math_verify
grading) on all 1319 questions through run_sgl_eval, at temperature 1.0 and top_p
1.0: the server's defaults for this checkpoint, which take the vocab-shard sampling
accept that FLy rewrites. The seed pins the sampling.

Registry: nightly-amd-4-gpu-mi35x-deepseek-v41-flash suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Server startup dominates, as in test_deepseek_v41_flash_eval_mi35x.py.
register_amd_ci(
    est_time=1200, suite="nightly-amd-4-gpu-mi35x-deepseek-v41-flash", nightly=True
)

DEEPSEEK_V41_FLASH_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V41_FLASH_MODEL_PATH", "deepseek-ai/DeepSeek-V4.1-Flash"
)
SERVER_LAUNCH_TIMEOUT = 5400
TP_SIZE = 4
FLY_WINDOW_SIZE = 2
FLY_ENTROPY_THRESHOLD = 0.3
# Measured 0.9659 on all 1319 questions with this setup on MI35x (accept length
# 4.51). The threshold sits about 0.05 under that, this repo's convention for
# sgl-eval gsm8k.
ACCURACY_THRESHOLD = 0.91


class TestDeepseekV41FlashFlyEvalMI35x(CustomTestCase):
    """DeepSeek-V4.1-Flash GSM8K accuracy with DSpark + FLy on AMD MI35x."""

    def test_deepseek_v41_flash_fly_gsm8k_accuracy(self):
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
            "--speculative-use-fly",
            "--speculative-fly-window-size",
            str(FLY_WINDOW_SIZE),
            "--speculative-fly-entropy-threshold",
            str(FLY_ENTROPY_THRESHOLD),
            "--cuda-graph-max-bs-decode",
            "256",
            "--max-running-requests",
            "256",
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
        # Same as test_deepseek_v41_flash_eval_mi35x.py: repeatable FlyDSL MoE
        # reduce, and no BF16-activation MoE route for these experts.
        env["AITER_FLYDSL_FORCE_REDUCE"] = "1"
        env["AITER_BF16_FP8_MOE_BOUND"] = "0"

        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            DEEPSEEK_V41_FLASH_MODEL_PATH,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )
        try:
            info = requests.get(base_url + "/server_info", timeout=60).json()
            self.assertTrue(info["speculative_use_fly"])
            self.assertEqual(info["speculative_fly_window_size"], FLY_WINDOW_SIZE)

            requests.get(base_url + "/flush_cache", timeout=60)
            metrics = run_sgl_eval(
                SimpleNamespace(
                    base_url=base_url,
                    model=DEEPSEEK_V41_FLASH_MODEL_PATH,
                    eval_name="gsm8k",
                    num_examples=None,
                    num_threads=128,
                    temperature=1.0,
                    top_p=1.0,
                    seed=42,
                )
            )
            states = requests.get(base_url + "/server_info", timeout=60).json()
            accept_length = states["internal_states"][0].get("avg_spec_accept_length")
        finally:
            kill_process_tree(process.pid)

        score = metrics["score"]
        passed = score >= ACCURACY_THRESHOLD
        status = "✅ PASS" if passed else "❌ FAIL"
        print(
            f"  gsm8k score={score:.4f} accept_length={accept_length} "
            f"threshold={ACCURACY_THRESHOLD} {status}"
        )
        if is_in_ci():
            summary = "### DeepSeek-V4.1-Flash + DSpark FLy (MI35x)\n\n"
            summary += (
                "| Model | TP | FLy | Accuracy | Accept length | Threshold | Status |\n"
            )
            summary += (
                "| ----- | -- | --- | -------- | ------------- | --------- | ------ |\n"
            )
            summary += (
                f"| {DEEPSEEK_V41_FLASH_MODEL_PATH} | {TP_SIZE} | "
                f"W{FLY_WINDOW_SIZE}/{FLY_ENTROPY_THRESHOLD} | {score:.3f} | "
                f"{accept_length} | {ACCURACY_THRESHOLD} | {status} |\n"
            )
            write_github_step_summary(summary)

        self.assertGreaterEqual(
            score,
            ACCURACY_THRESHOLD,
            f"DeepSeek-V4.1-Flash + FLy gsm8k accuracy {score:.3f} below threshold "
            f"{ACCURACY_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
