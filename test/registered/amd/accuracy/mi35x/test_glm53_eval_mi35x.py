"""MI35x GLM-5.3 GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.3 (FP8) on MI35x (gfx950) with the GLM-5.3 cookbook's
MI355X / FP8 / low-latency / single-node cell: TP8, DSA tilelang prefill+decode,
131072 chunked prefill, 0.80 static memory fraction, and a 20-minute watchdog
for weight loading. No MTP: the cookbook disables speculative decoding on AMD.

The day-0 dashboard (https://rocm.github.io/sglang-ci/day0/) lists MI355X FP8 as
a recipe cell with no CI behind it. The cookbook ships this cell as
`verified: false` with no AMD benchmark numbers, so this nightly is also the
first recurring measurement of it.

GLM-5.3 keeps the GLM-5.2 base architecture (`glm_moe_dsa`, 78 layers, 256
routed experts) and ships as FP8, so it runs into the hazard the GLM-5.2-FP8
nightly used to guard against on gfx950, and it replaces that job in the
workflow: an FP8 GEMM error that is small per layer but compounds across 78
layers collapses GSM8K while short prompts still look fine. Only a multi-step
reasoning eval catches that class of regression.

Measured on the full GSM8K split: 0.9742 on rocm724 (1370 s wall clock, 576 s
weight load including the download into the pool's shared cache), 0.9712 on
rocm10 (1018 s) and 0.9742 on rocm720 (1105 s) against the warm cache. HF
snapshot aca966e4e02791568aa6a4ced368624b3d897f42, runs 36523773291,
36541623736 and 36541643777.

Eval harness: sgl-eval's gsm8k (zero-shot chat, \\boxed{} extraction,
math_verify grading) through run_sgl_eval, rather than the legacy few-shot
scorer that run_combined_tests routes gsm8k to. GLM-5.3's chat template
forces thinking on, and the legacy scorer takes the last number in the
response, which a reasoning trace makes meaningless. The sampling parameters
match the GLM-5.3-Flash nightlies: 64 threads, 32768 max tokens, temperature
1.0, top_p 0.95 (the checkpoint's generation defaults), thinking on. The seed
pins the sampling so a failure is a regression rather than a reroll.

Threshold: the cookbook's CUDA cells measure 97.12-97.73% on GSM8K. 0.92
follows this repo's `measured - 0.05` convention for sgl-eval gsm8k thresholds
and matches the GLM-5.2-FP8 and GLM-5.3-Flash tests on AMD.

Registry: nightly-amd-8-gpu-mi35x-glm53 suite
"""

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
    ModelLaunchSettings,
    popen_launch_server,
)

# Register for AMD CI - MI35x GLM-5.3 accuracy test (~2 h: the ~700 GB FP8
# checkpoint dominates startup, then a full-split thinking GSM8K at TP8)
register_amd_ci(
    est_time=7200,
    suite="nightly-amd-8-gpu-mi35x-glm53",
    nightly=True,
)

GLM_53_MODEL_PATH = "zai-org/GLM-5.3"
BASELINE_ACCURACY = 0.92

# Fetching and loading a ~700 GB checkpoint against a cold cache is what this
# budget has to cover; the default launch timeout is nowhere near enough.
SERVER_LAUNCH_TIMEOUT = 5400


class TestGLM53EvalMI35x(unittest.TestCase):
    """GLM-5.3 GSM8K Accuracy Evaluation Test for MI35x."""

    def test_glm_53(self):
        """Run accuracy test for GLM-5.3."""
        cookbook_args = [
            "--trust-remote-code",
            "--reasoning-parser=glm45",
            "--tool-call-parser=glm47",
            "--dsa-prefill-backend=tilelang",
            "--dsa-decode-backend=tilelang",
            "--chunked-prefill-size=131072",
            "--mem-fraction-static=0.80",
            "--watchdog-timeout=1200",
            # Not part of the cookbook cell; purely a load-time win on a
            # checkpoint this large, with no effect on numerics.
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        model = ModelLaunchSettings(
            GLM_53_MODEL_PATH,
            tp_size=8,
            extra_args=cookbook_args,
            env={"SGLANG_USE_AITER": "1"},
            variant="TP8",
        )

        # run_combined_tests routes gsm8k to the legacy scorer, so launch the
        # server here and hand the eval to sgl-eval directly.
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model.model_path,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=model.extra_args,
            env=model.env,
        )
        try:
            metrics = run_sgl_eval(
                SimpleNamespace(
                    base_url=base_url,
                    model=model.model_path,
                    eval_name="gsm8k",
                    num_examples=None,
                    num_threads=64,
                    max_tokens=32768,
                    temperature=1.0,
                    top_p=0.95,
                    seed=42,
                    sgl_eval_thinking=True,
                )
            )
        finally:
            kill_process_tree(process.pid)

        score = metrics["score"]
        passed = score >= BASELINE_ACCURACY
        write_accuracy_github_summary(
            "GLM-5.3 (MI35x)",
            "gsm8k",
            [
                AccuracyTestResult(
                    model=model.model_path,
                    dataset="gsm8k",
                    passed=passed,
                    score=score,
                    baseline_accuracy=BASELINE_ACCURACY,
                    error=None if passed else "below baseline",
                    latency=metrics.get("latency"),
                    variant=model.variant,
                )
            ],
        )
        self.assertGreaterEqual(score, BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
