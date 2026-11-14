"""MI35x Muse Spark Open MXFP4 MMMU Accuracy Evaluation Test (8-GPU)

Gates the image path of amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 on MI35x
(gfx950). Muse Spark Open is multimodal, and the GSM8K nightly
(test_muse_spark_mxfp4_eval_mi35x.py) only sends text, so the vision encoder
and the image-embedding path through the ROCm model (rocm_muse_spark) have no
other coverage on AMD.

The server is launched with exactly the arguments and environment of that
GSM8K test: the cookbook's MI355X TP8 / EP8 command with aiter attention, FP8
KV cache and DFlash2, on the default Triton MXFP4 expert backend. A failure
here and a pass there therefore points at the image path, not at the config.

Eval harness: run_eval's mmmu (100 fixed samples across all MMMU subjects,
"Answer: X" extraction), with thinking at reasoning effort xhigh and the same
sampling as the GSM8K test. The muse reasoning parser keeps the trace in
`reasoning_content`, so the answer is read from `content`.

Registry: nightly-amd-accuracy-8-gpu-mi35x-muse-spark-mmmu suite
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
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-accuracy-8-gpu-mi35x-muse-spark-mmmu",
    nightly=True,
)

MUSE_SPARK_MODEL_PATH = os.environ.get(
    "MUSE_SPARK_MODEL_PATH", "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
)
MUSE_SPARK_DRAFT_PATH = os.environ.get(
    "MUSE_SPARK_DRAFT_PATH", "RadixArk/Muse-Spark-Open-DFlash2"
)
SERVER_LAUNCH_TIMEOUT = 9000
# Provisional: there is no AMD MMMU measurement for this model yet. Replace
# with the measured score minus 0.05 once the first run reports one.
BASELINE_ACCURACY = 0.50
TP_SIZE = 8
MAX_RUNNING_REQUESTS = 64

# Same as test_muse_spark_mxfp4_eval_mi35x.py, so the two gates share a config.
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


class TestMuseSparkMXFP4MMMUEvalMI35x(CustomTestCase):
    """Muse Spark Open MXFP4 MMMU Accuracy Evaluation Test for AMD MI35x."""

    def test_muse_spark_mxfp4_mmmu_accuracy(self):
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            MUSE_SPARK_MODEL_PATH,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=COOKBOOK_ARGS,
            env={**os.environ, **COOKBOOK_ENV},
        )
        try:
            metrics = run_eval(
                SimpleNamespace(
                    base_url=base_url,
                    model=MUSE_SPARK_MODEL_PATH,
                    eval_name="mmmu",
                    num_examples=100,
                    num_threads=MAX_RUNNING_REQUESTS,
                    max_tokens=65536,
                    temperature=1.0,
                    top_p=1.0,
                    chat_template_kwargs={"reasoning_effort": "xhigh"},
                )
            )
        finally:
            kill_process_tree(process.pid)

        score = metrics["score"]
        passed = score >= BASELINE_ACCURACY
        write_accuracy_github_summary(
            "Muse Spark Open MXFP4 (MI35x)",
            "mmmu",
            [
                AccuracyTestResult(
                    model=MUSE_SPARK_MODEL_PATH,
                    dataset="mmmu",
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
