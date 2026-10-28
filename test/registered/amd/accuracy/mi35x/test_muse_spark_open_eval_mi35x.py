"""MI35x Muse Spark Open MXFP4 GSM8K + MMMU Accuracy Test (4-GPU, TP=4)

Tests amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4, AMD's Quark MXFP4
quantization of Meta's Muse Spark Open (816B total / ~42B active multimodal
MoE), with the cookbook's MI355X / MXFP4 / Low Latency recipe: TP4/EP4, aiter
attention at page size 16, FlyDSL experts, DFlash2 speculative decoding with a
Triton draft-attention backend, and the muse reasoning and tool-call parsers.

On HIP, ``sglang.srt.models.muse_spark`` dispatches to the separate ROCm
implementation in ``sglang.srt.rocm_muse_spark``, and the recipe depends on
aiter patches from ``3rdparty/amd/muse_spark`` (a CK batch-prefill VM-fault fix
and the FlyDSL Muse MoE kernels) that the Day-0 image bakes in. The workflow
job applies them to the CI image before this file runs, so a nightly also
catches the patches falling out of step with the pinned aiter.

One server serves both evals, so they gate the exact configuration users
deploy:

  * GSM8K, through sgl-eval (zero-shot chat, thinking on, reasoning_effort
    xhigh, 32768 max tokens, temperature 1.0, top_p 0.95). The reasoning parser
    keeps the trace out of ``content``. These are the settings and the 0.93
    gate of the B300 Muse Spark GSM8K test on the same checkpoint family.
  * MMMU (100 samples), because Muse Spark Open is multimodal and the vision
    path through the ROCm implementation is otherwise untested.

MXFP4 needs gfx95x, so this is MI35x-only.

Registry: nightly-amd-4-gpu-mi35x-muse-spark-open suite
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.accuracy_test_runner import (
    AccuracyTestResult,
    write_accuracy_github_summary,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD CI - MI35x Muse Spark Open MXFP4 accuracy (~2 h: loading the
# ~450 GB checkpoint, then a full-split thinking GSM8K and 100 MMMU samples)
register_amd_ci(
    est_time=7200,
    suite="nightly-amd-4-gpu-mi35x-muse-spark-open",
    nightly=True,
)

MODEL_PATH = "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
DRAFT_MODEL_PATH = "RadixArk/Muse-Spark-Open-DFlash2"
TP_SIZE = 4
VARIANT = "TP4+EP4+MXFP4+aiterAttn+FlyDSL+DFlash2"

GSM8K_THRESHOLD = 0.93
# Provisional: there is no AMD MMMU measurement for this model yet. Replace
# with the measured score minus 0.05 once the first nightly reports one.
MMMU_THRESHOLD = 0.50

# Fetching and loading a ~450 GB checkpoint against a cold cache is what this
# budget has to cover.
SERVER_LAUNCH_TIMEOUT = 5400

# The cookbook's MI355X / MXFP4 / Low Latency cell. No --quantization: the
# Quark MXFP4 checkpoint is detected from its config.
SERVER_ARGS = [
    "--tp-size",
    str(TP_SIZE),
    "--ep-size",
    str(TP_SIZE),
    "--attention-backend",
    "aiter",
    "--page-size",
    "16",
    "--mem-fraction-static",
    "0.85",
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-model-path",
    DRAFT_MODEL_PATH,
    "--speculative-draft-attention-backend",
    "triton",
    "--speculative-num-draft-tokens",
    "8",
    "--reasoning-parser",
    "muse",
    "--tool-call-parser",
    "muse",
    # Not part of the cookbook cell; covers weight loading on a checkpoint
    # this large, with no effect on numerics.
    "--watchdog-timeout",
    "1200",
]
SERVER_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "MUSE_MOE_BACKEND": "flydsl",
}


class TestMuseSparkOpenMxfp4EvalMI35x(unittest.TestCase):
    """Muse Spark Open MXFP4 GSM8K + MMMU accuracy on AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=SERVER_ARGS,
            env=SERVER_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _record(self, dataset, score, threshold, latency=None):
        passed = score >= threshold
        write_accuracy_github_summary(
            "Muse Spark Open MXFP4 (MI35x)",
            dataset,
            [
                AccuracyTestResult(
                    model=MODEL_PATH,
                    dataset=dataset,
                    passed=passed,
                    score=score,
                    baseline_accuracy=threshold,
                    error=None if passed else "below baseline",
                    latency=latency,
                    variant=VARIANT,
                )
            ],
        )
        self.assertGreaterEqual(score, threshold)

    def _report_accept_length(self):
        # Reported, not gated: no AMD measurement exists yet to set a floor.
        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        response.raise_for_status()
        states = response.json().get("internal_states") or [{}]
        accept_length = states[0].get("avg_spec_accept_length")
        print(f"avg_spec_accept_length={accept_length}")
        if is_in_ci() and accept_length is not None:
            write_github_step_summary(
                f"DFlash2 avg_spec_accept_length: {accept_length:.4f}\n"
            )

    def test_a_gsm8k(self):
        """Full-split GSM8K through sgl-eval with thinking on."""
        requests.get(self.base_url + "/flush_cache", timeout=60)
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=MODEL_PATH,
                eval_name="gsm8k",
                num_examples=None,
                num_threads=64,
                max_tokens=32768,
                temperature=1.0,
                top_p=0.95,
                seed=42,
                sgl_eval_thinking=True,
                reasoning_effort="xhigh",
            )
        )
        self._report_accept_length()
        self._record("gsm8k", metrics["score"], GSM8K_THRESHOLD, metrics.get("latency"))

    def test_b_mmmu(self):
        """100-sample MMMU, so the image path is exercised on ROCm."""
        requests.get(self.base_url + "/flush_cache", timeout=60)
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=MODEL_PATH,
                eval_name="mmmu",
                num_examples=100,
                num_threads=64,
                max_tokens=32768,
                reasoning_effort="xhigh",
            )
        )
        self._record("mmmu", metrics["score"], MMMU_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
