"""MI30x GLM-5.3 GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.3 (FP8) on MI30x (gfx942) with the GLM-5.3 cookbook's
MI300X / FP8 / low-latency / single-node cell: TP8, DSA tilelang prefill+decode,
131072 chunked prefill, 0.80 static memory fraction, and a 20-minute watchdog
for weight loading. The MI325X cell uses the same flags. Same eval and
threshold as the gfx950 gate in test_glm53_eval_mi35x.py.

The day-0 dashboard (https://rocm.github.io/sglang-ci/day0/) lists MI30x FP8 as
a recipe cell with no CI behind it. gfx942 is not redundant with gfx950 here:
it uses the FP8 e4m3fnuz format rather than OCP e4m3, so the FP8 weights are
converted at load time, and it takes different AITER and tilelang kernels for
the same forward pass. The MI35x job covers only gfx950.

MI300X (192 GB x 8) holds the ~700 GB FP8 weights with room for KV cache; the
BF16 checkpoint does not fit single-node here, which is why this tests FP8.

Measured on the full GSM8K split: 0.9735 on rocm724 (4332 s wall clock) and
0.9704 on rocm720 (4565 s). Weight load took 2892 s and 3016 s, most of it
waiting on the first download into the pool's shared cache. HF snapshot
aca966e4e02791568aa6a4ced368624b3d897f42, runs 36505808540 and 36505823619.
Both sit inside the cookbook's CUDA range of 97.12-97.73%.

Eval harness: sgl-eval's gsm8k through run_sgl_eval, rather than the legacy
few-shot scorer that run_combined_tests routes gsm8k to, because GLM-5.3's
chat template forces thinking on and the legacy scorer reads the last number
in the response. See the MI35x file for the longer note.

Registry: nightly-amd-accuracy-8-gpu-glm53 suite
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

# Register for AMD CI - MI30x GLM-5.3 accuracy test. Measured 4332 s on
# rocm724 and 4565 s on rocm720 against a cold shared cache; 9000 s leaves
# room for a slower download of the ~700 GB checkpoint.
register_amd_ci(
    est_time=9000,
    suite="nightly-amd-accuracy-8-gpu-glm53",
    nightly=True,
)

GLM_53_MODEL_PATH = "zai-org/GLM-5.3"
BASELINE_ACCURACY = 0.92

# Fetching and loading a ~700 GB checkpoint against a cold cache is what this
# budget has to cover; the default launch timeout is nowhere near enough.
SERVER_LAUNCH_TIMEOUT = 7200


class TestGLM53EvalMI30x(unittest.TestCase):
    """GLM-5.3 GSM8K Accuracy Evaluation Test for MI30x."""

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
            "GLM-5.3 (MI30x)",
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
