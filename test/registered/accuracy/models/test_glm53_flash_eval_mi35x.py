"""MI35x GLM-5.3-Flash GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.3-Flash on MI35x (gfx950) with the AMD FP8 recipe from the
GLM-5.3-Flash cookbook refresh (#36712): TP8 + EP8, BF16 KV cache, TileLang DSA
prefill+decode, Triton linear attention, AITER MoE runner, SGLANG_USE_AITER=1,
full decode graphs at batch sizes 1 and 32.

GLM-5.3-Flash is the first GLM checkpoint whose 45 text layers mix three
attention kinds -- MLA, DSA sparse, and KDA linear -- behind mHC residuals, so
it exercises engine paths no other AMD nightly covers. The KDA state pool is a
second memory pool alongside the paged KV pool, and the mHC pre/post ops sit on
every layer boundary. A single-arch gate would not be enough: gfx950 takes the
AITER MoE runner, which applies the SwiGLU clamp itself, and the AITER mHC
pre/post, while gfx942 takes the Triton MoE runner and the generic mHC path.
This file gates the gfx950 half; the gfx942 half is
test_glm53_flash_eval_mi30x.py.

Threshold: #36607 measured the full 1319-question GSM8K split at
1288/1319 = 97.65% on MI355X (97.35% on MI300X). 0.92 follows this repo's
`measured - 0.05` convention for sgl-eval gsm8k thresholds, which also leaves
room for the sampling noise the checkpoint's own generation defaults introduce.
It lands on the same 0.92 as the GLM-5.2-FP8 nightlies on both AMD and CUDA, so
a red run here reads as "GLM-5.3-Flash on gfx950 regressed" rather than "this
gate is stricter than its neighbours". The cookbook lists this exact MI355X
command as having passed a runtime pilot with full GSM8K still pending, so this
job is also that measurement.

This harness reproduced #36607's 97.65% to within 0.006 on the GLM-5.3-Flash
support branch (0.9704 on rocm720, 0.9712 on rocm724, TP8 with graphs off), so
the sgl-eval zero-shot/\\boxed{}/math_verify path and the checkpoint's sampling
defaults are not costing accuracy relative to #36607's harness.

Eval harness: `api="sgl_eval"` rather than the default 5-shot completion
scorer. GLM-5.3-Flash thinks by default, and the completion scorer takes the
last number in the response, which a reasoning trace makes meaningless. The
sgl-eval path is zero-shot chat with \\boxed{} extraction and math_verify
grading, and the parameters below are the accuracy command the cookbook
publishes for this model: 64 threads, 32768 max tokens, temperature 1.0,
top_p 0.95, thinking on. The seed pins the sampling so a failure is a
regression rather than a reroll.

Registry: nightly-amd-8-gpu-mi35x-glm53-flash suite
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Register for AMD CI - MI35x GLM-5.3-Flash accuracy test. Measured 3322 s
# on rocm720 and 2673 s on rocm724; 5400 s covers a cold-cache load of the
# 328 GB checkpoint plus the eval.
register_amd_ci(
    est_time=5400,
    suite="nightly-amd-8-gpu-mi35x-glm53-flash",
    nightly=True,
)

GLM_53_FLASH_MODEL_PATH = "zai-org/GLM-5.3-Flash"

# Fetching and loading a 328 GB checkpoint against a cold cache is what this
# budget has to cover; the default launch timeout is nowhere near enough.
SERVER_LAUNCH_TIMEOUT = 5400


class TestGLM53FlashEvalMI35x(unittest.TestCase):
    """GLM-5.3-Flash GSM8K Accuracy Evaluation Test for MI35x."""

    def test_glm_53_flash(self):
        """Run accuracy test for GLM-5.3-Flash."""
        cookbook_args = [
            "--ep-size=8",
            "--attention-backend=dsa",
            "--dsa-prefill-backend=tilelang",
            "--dsa-decode-backend=tilelang",
            "--linear-attn-backend=triton",
            "--kv-cache-dtype=bfloat16",
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

        variants = [
            ModelLaunchSettings(
                GLM_53_FLASH_MODEL_PATH,
                tp_size=8,
                extra_args=cookbook_args,
                env={"SGLANG_USE_AITER": "1"},
                variant="TP8-EP8",
                launch_timeout=SERVER_LAUNCH_TIMEOUT,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-5.3-Flash (MI35x)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.92,
                api="sgl_eval",
                num_threads=64,
                max_tokens=32768,
                temperature=1.0,
                top_p=0.95,
                seed=42,
                sgl_eval_thinking=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
