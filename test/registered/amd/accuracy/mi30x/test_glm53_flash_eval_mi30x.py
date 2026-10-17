"""MI30x GLM-5.3-Flash GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.3-Flash on MI30x (gfx942) with the AMD FP8 recipe from the
GLM-5.3-Flash cookbook refresh (#36712): TP8 + EP8, BF16 KV cache, TileLang DSA
prefill+decode, Triton linear attention, SGLANG_USE_AITER=1, full decode graphs
at batch sizes 1 and 32. Same eval and threshold as the gfx950 gate in
test_glm53_flash_eval_mi35x.py.

MoE runner: this deviates from the cookbook cell, which names the Triton runner
for MI300X. On current main the Triton MoE runner makes this model generate
without ever stopping on gfx942: every sequence runs to the token cap and
scores zero. Measured in one job that ran three 64-question evals back to back
on the same server image (run 36119287477, 2026-09-26): Triton MoE with the
default fused sgl-kernel top-k scored 0/64 at 2048+ tokens per sequence,
Triton MoE with #36607's portable Torch top-k also scored 0/64 at 2048+ tokens
per sequence, and the AITER MoE runner scored 63/64 at 151 tokens per sequence.
The DSA top-k backend makes no difference, so the fault is the Triton MoE
runner itself, and the cookbook's MI300X MoE recommendation is stale.

gfx942 is not redundant with gfx950 for this model. It runs the generic mHC
path, since AITER mHC is gfx95-only, and nothing else gives that path nightly
coverage for this model. That path reaches gfx942 only with the HIP guard in
#41136: without it, TileLang's HIP codegen cannot lower the tl.get_lane_idx in
the fused mHC post/pre kernel and decode graph capture dies with "Unresolved
call Op(tl.get_lane_idx)" (run 36079282524).

Measured on main plus both HIP fixes in #41136, which this test requires:
0.9750 (1286/1319) on the rocm10 image, with a 2769 s weight load, a 1391 s
eval and 4419 s of wall clock (run 36232707853). That is the same count the
gfx950 gate in test_glm53_flash_eval_mi35x.py scored, so the gfx942 fallback
paths are not costing accuracy relative to the gfx950 fast paths.

Threshold: 0.92 follows this repo's `measured - 0.05` convention for sgl-eval
gsm8k thresholds and matches the gfx950 gate, so the two arches stay directly
comparable.

Runtime: the 328 GB checkpoint has taken 2769-4650 s to load from this pool's
shared cache, and the eval 1333-2374 s on top of that. In run 36678520753 the
first launch on all three images was still loading at 5400 s and only the CI's
online retry brought the server up, so the launch timeout below is 9000 s. The
workflow allows 18000 s. If that ever proves tight, prefer raising it over
trimming the eval: a full-split score is what makes this arch's number
comparable to the gfx950 one.

Eval harness: sgl-eval's gsm8k through run_sgl_eval, rather than the legacy
few-shot scorer that run_combined_tests routes gsm8k to, because
GLM-5.3-Flash thinks by default and the legacy scorer reads the last number
in the response. The parameters below are the accuracy command the cookbook
publishes for this model. See the MI35x file for the longer note.

Registry: nightly-amd-accuracy-8-gpu-glm53-flash suite
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

# Register for AMD CI - MI30x GLM-5.3-Flash accuracy test. The 9000 s launch
# budget below plus the slowest full-split eval measured (2374 s).
register_amd_ci(
    est_time=11400,
    suite="nightly-amd-accuracy-8-gpu-glm53-flash",
    nightly=True,
)

GLM_53_FLASH_MODEL_PATH = "zai-org/GLM-5.3-Flash"
BASELINE_ACCURACY = 0.92

# Fetching and loading a 328 GB checkpoint against a cold cache is what this
# budget has to cover; the default launch timeout is nowhere near enough.
# Loads have measured up to 4650 s, and a first launch has run past 5400 s.
SERVER_LAUNCH_TIMEOUT = 9000


class TestGLM53FlashEvalMI30x(unittest.TestCase):
    """GLM-5.3-Flash GSM8K Accuracy Evaluation Test for MI30x."""

    def test_glm_53_flash(self):
        """Run accuracy test for GLM-5.3-Flash."""
        cookbook_args = [
            "--ep-size=8",
            "--attention-backend=dsa",
            "--dsa-prefill-backend=tilelang",
            "--dsa-decode-backend=tilelang",
            "--linear-attn-backend=triton",
            "--kv-cache-dtype=bfloat16",
            # Not the cookbook's Triton runner; see the MoE runner note above.
            "--moe-runner-backend=aiter",
            "--cuda-graph-backend-decode=full",
            "--cuda-graph-backend-prefill=disabled",
            "--cuda-graph-bs-decode",
            "1",
            "32",
            "--reasoning-parser=glm45",
            "--tool-call-parser=glm47",
            "--watchdog-timeout=1200",
            # Not part of the cookbook cell; purely a load-time win on a
            # checkpoint this large, with no effect on numerics.
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]

        model = ModelLaunchSettings(
            GLM_53_FLASH_MODEL_PATH,
            tp_size=8,
            extra_args=cookbook_args,
            env={"SGLANG_USE_AITER": "1"},
            variant="TP8-EP8",
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
            "GLM-5.3-Flash (MI30x)",
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
