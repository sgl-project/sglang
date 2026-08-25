"""MI35x Kimi-K3 GSM8K Completion Evaluation Test (8-GPU)

Tests moonshotai/Kimi-K3 with GSM8K few-shot benchmark on MI35x.

Server arguments follow the Day-0 recipe in the AMD tracking issue
(sgl-project/sglang#32548) for the non-speculative config: TP8 with the
Triton attention backend, the AITER FlyDSL A8W4 SiTU MoE path, and the radix
cache disabled.

That issue reports throughput on MI355 TP8 but no accuracy, and the cookbook
cell for this topology is still published as `verified: false`. So the recipe
is known to run at speed; what is missing, and what this test supplies, is
evidence that it produces correct tokens.

First green run on 8xMI355X scored 0.956 with 0.2% unparsable, against the
0.92 threshold, in about 74 minutes end to end.

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
# The one deviation from the tracking-issue recipe, which leaves concurrency
# unbounded and captures decode graphs to 256. Those runs drove bounded
# concurrency (<= 32); this eval submits every question at once, so it needs an
# explicit ceiling to stay inside the ~53 GB per GPU left after weights.
# --cuda-graph-max-bs-decode is kept equal to it, since capture across K3's 93
# attention + 92 MoE layers is expensive and capturing above the concurrency
# ceiling buys nothing.
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
            "--disable-radix-cache",
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
        # AITER supplies K3's MoE on ROCm. SGLANG_AITER_K3_OPT is read in
        # mxfp4.py and models/kimi_k3.py but is gated behind SGLANG_USE_AITER,
        # so both must be set. AITER_SITUV2_A8W4 selects the W4A8 SiTU expert
        # kernels; it additionally requires the MoE activation to be "situ",
        # which K3 supplies from its own config. AITER_FLYDSL_FORCE is consumed
        # by AITER itself rather than sglang -- it will not grep to anything
        # in-tree.
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
