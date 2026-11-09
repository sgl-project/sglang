"""MI35x Muse Spark Open MXFP4 GSM8K Accuracy Evaluation Test (8-GPU)

Tests amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 on MI35x (gfx950) with the
cookbook's MI355X TP8 / EP8 command: aiter attention, FP8 KV cache, DFlash2
speculative decoding with real draft acceptance (no SGLANG_SIMULATE_*), and the
ROCm Muse Spark model path (rocm_muse_spark) with its default Triton MXFP4
expert backend.

Deviations from the cookbook command, each on purpose:
- --max-running-requests and --cuda-graph-max-bs-decode are 64, matching the
  eval's threads, so the full GSM8K split finishes inside the job.
- --watchdog-timeout 1800 so loading the ~470 GB checkpoint is not mistaken
  for a hang.

Eval harness: sgl-eval's gsm8k (zero-shot chat, math_verify grading) through
run_sgl_eval, on the full split with thinking on at reasoning effort xhigh
(the day0 accuracy protocol). The muse reasoning parser
keeps the trace in `reasoning_content` and the answer in `content`.
Temperature 1.0 and top_p 1.0 with a fixed seed: greedy decoding lets the
thinking loop until truncation.

Registry: nightly-amd-accuracy-8-gpu-mi35x-muse-spark suite
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.accuracy_test_runner import (
    AccuracyTestResult,
    write_accuracy_github_summary,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-accuracy-8-gpu-mi35x-muse-spark",
    nightly=True,
)

MUSE_SPARK_MODEL_PATH = os.environ.get(
    "MUSE_SPARK_MODEL_PATH", "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
)
MUSE_SPARK_DRAFT_PATH = os.environ.get(
    "MUSE_SPARK_DRAFT_PATH", "RadixArk/Muse-Spark-Open-DFlash2"
)
SERVER_LAUNCH_TIMEOUT = 9000
# Measured 0.9757 at TP8 (sgl-eval gsm8k, real acceptance); measured - 0.05, as
# for the other sgl-eval gsm8k gates.
BASELINE_ACCURACY = 0.92
TP_SIZE = 8
MAX_RUNNING_REQUESTS = 64

COOKBOOK_ARGS = [
    "--tp-size",
    str(TP_SIZE),
    "--ep-size",
    str(TP_SIZE),
    "--attention-backend",
    "aiter",
    "--page-size",
    "16",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--mem-fraction-static",
    "0.85",
    "--max-running-requests",
    str(MAX_RUNNING_REQUESTS),
    "--cuda-graph-max-bs-decode",
    str(MAX_RUNNING_REQUESTS),
    "--swa-full-tokens-ratio",
    "0.2",
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-model-path",
    MUSE_SPARK_DRAFT_PATH,
    "--speculative-draft-attention-backend",
    "triton",
    "--speculative-num-draft-tokens",
    "8",
    "--speculative-draft-window-size",
    "2048",
    "--reasoning-parser",
    "muse",
    "--tool-call-parser",
    "muse",
    "--watchdog-timeout",
    "1800",
]

COOKBOOK_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
}


class TestMuseSparkMXFP4EvalMI35x(CustomTestCase):
    """Muse Spark Open MXFP4 GSM8K Accuracy Evaluation Test for AMD MI35x."""

    def test_muse_spark_mxfp4_gsm8k_accuracy(self):
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            MUSE_SPARK_MODEL_PATH,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=COOKBOOK_ARGS,
            env={**os.environ, **COOKBOOK_ENV},
        )
        try:
            metrics = run_sgl_eval(
                SimpleNamespace(
                    base_url=base_url,
                    model=MUSE_SPARK_MODEL_PATH,
                    eval_name="gsm8k",
                    num_examples=None,
                    num_threads=MAX_RUNNING_REQUESTS,
                    max_tokens=65536,
                    temperature=1.0,
                    top_p=1.0,
                    seed=42,
                    sgl_eval_thinking=True,
                    chat_template_kwargs={"reasoning_effort": "xhigh"},
                )
            )
        finally:
            kill_process_tree(process.pid)

        score = metrics["score"]
        passed = score >= BASELINE_ACCURACY
        write_accuracy_github_summary(
            "Muse Spark Open MXFP4 (MI35x)",
            "gsm8k",
            [
                AccuracyTestResult(
                    model=MUSE_SPARK_MODEL_PATH,
                    dataset="gsm8k",
                    passed=passed,
                    score=score,
                    baseline_accuracy=BASELINE_ACCURACY,
                    error=None if passed else "below baseline",
                    latency=metrics.get("latency"),
                    variant=f"TP{TP_SIZE} DFLASH",
                )
            ],
        )
        self.assertGreaterEqual(score, BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
