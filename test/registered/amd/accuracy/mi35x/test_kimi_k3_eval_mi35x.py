"""MI35x Kimi-K3 GSM8K Completion Evaluation Test (8-GPU)

Tests moonshotai/Kimi-K3 with GSM8K few-shot benchmark on MI35x.

Server arguments follow the Kimi-K3 cookbook's MI350X/MI355X 1x8 cell
(Unified / Balanced): TP8 with the Triton attention backend and the AITER
FlyDSL A8W4 SiTU MoE path. That cell is published as `verified: false` and
carries no benchmark entry, so this is the first end-to-end K3 signal on
ROCm.

K3's native MXFP4 weights need gfx95x, so this runs on MI35x only -- mxfp4
does not register on gfx942 (MI300/MI325). The 2.8T checkpoint is ~1.5 TB in
MXFP4, roughly 192 GB of the 288 GB on each of the 8 GPUs, which is why
`--mem-fraction-static` stays at the cookbook's 0.85 and concurrency is
capped rather than left unbounded.

The eval uses the few-shot *completion* harness that every sibling Kimi MI35x
test uses (K2, K2.5, K2.6). It deliberately bypasses the chat template: K3
has thinking permanently enabled and routes its answer through
`reasoning_content` on the chat path, which would leave `message.content`
empty and score 0. Scoring raw completions keeps this test measuring what it
is meant to measure -- whether the ROCm kernels produce correct tokens.

Registry: nightly-amd-accuracy-8-gpu-mi35x-kimi-k3 suite
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

# Register for AMD CI - Kimi K3 accuracy test on MI35x (~150 min: the 1.5 TB
# MXFP4 checkpoint dominates startup, then 1319 GSM8K questions at TP8)
register_amd_ci(
    est_time=9000, suite="nightly-amd-accuracy-8-gpu-mi35x-kimi-k3", nightly=True
)

KIMI_K3_MODEL_PATH = os.environ.get("KIMI_K3_MODEL_PATH", "moonshotai/Kimi-K3")
SERVER_LAUNCH_TIMEOUT = 9000
ACCURACY_THRESHOLD = 0.92
TP_SIZE = 8
# Bounded so the KDA state pool and the MLA KV pool both stay well inside the
# ~53 GB per GPU left after weights. Kept equal to --cuda-graph-max-bs-decode:
# graph capture across K3's 93 attention + 92 MoE layers is expensive, and
# capturing above the concurrency ceiling buys nothing.
MAX_RUNNING_REQUESTS = 64


class TestKimiK3EvalMI35x(CustomTestCase):
    """Kimi-K3 GSM8K Completion Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
        cls.max_new_tokens = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "512"))

    def test_kimi_k3_gsm8k_accuracy(self):
        """Test Kimi-K3 with GSM8K few-shot completion benchmark."""
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--attention-backend",
            "triton",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.85",
            "--cuda-graph-max-bs-decode",
            str(MAX_RUNNING_REQUESTS),
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--reasoning-parser",
            "kimi_k3",
            "--tool-call-parser",
            "kimi_k3",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]
        env = os.environ.copy()
        # AITER supplies K3's MoE on ROCm: FlyDSL is the HIP counterpart of the
        # CuTe-DSL path, and SITUV2_A8W4 selects the W4A8 SiTU expert kernels.
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_AITER_K3_OPT"] = "1"
        env["AITER_FLYDSL_FORCE"] = "1"
        env["AITER_SITUV2_A8W4"] = "1"

        process = popen_launch_server(
            KIMI_K3_MODEL_PATH,
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
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            acc = metrics["accuracy"]

            passed = acc >= ACCURACY_THRESHOLD
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  accuracy={acc:.3f} threshold={ACCURACY_THRESHOLD} {status}")

            if is_in_ci():
                summary = "### Kimi-K3 Model (MI35x)\n\n"
                summary += "| Model | TP | Accuracy | Threshold | Status |\n"
                summary += "| ----- | -- | -------- | --------- | ------ |\n"
                summary += f"| {KIMI_K3_MODEL_PATH} | {TP_SIZE} | {acc:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                acc,
                ACCURACY_THRESHOLD,
                f"Kimi-K3 accuracy {acc:.3f} below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
