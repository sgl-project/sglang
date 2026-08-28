"""MI35x GLM-5.3-Flash GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.3-Flash on MI35x (gfx950) with the ROCm recipe the cookbook
ships as `verified: true`: TP8, BF16 KV cache, TileLang DSA prefill+decode,
Triton MoE runner, AITER on, CUDA graphs off.

GLM-5.3-Flash is the first GLM checkpoint whose 45 text layers mix three
attention kinds -- MLA, DSA sparse, and KDA linear -- behind mHC residuals, so
it exercises engine paths no other AMD nightly covers. The KDA state pool is a
second memory pool alongside the paged KV pool, and the mHC pre/post ops sit on
every layer boundary. A single-arch gate would not be enough here: gfx942 and
gfx950 run materially different kernels for the same model. gfx950 takes the
fused SGL-kernel k-pool top-k path and the AITER mHC pre/post plus fused
attention-to-FFN boundary, while gfx942 falls back to the portable unfused
Torch DSA top-k and the generic mHC path. This file gates the gfx950 half; the
gfx942 half is test_glm53_flash_eval_mi30x.py.

Threshold: AMD enablement (#36607) measured the full 1319-question GSM8K split
at 1288/1319 = 97.65% on MI355X (97.35% on MI300X). 0.92 follows this repo's
`measured - 0.05` convention for sgl-eval gsm8k thresholds, which also leaves
room for the sampling noise the checkpoint's own generation defaults introduce.
It lands on the same 0.92 as the GLM-5.2-FP8 nightlies on both AMD and CUDA, so
a red run here reads as "GLM-5.3-Flash on gfx950 regressed" rather than "this
gate is stricter than its neighbours".

Eval harness: `api="sgl_eval"` rather than the default 5-shot completion
scorer. GLM-5.3-Flash thinks by default, and the completion scorer takes the
last number in the response, which a reasoning trace makes meaningless. The
sgl-eval path is zero-shot chat with \\boxed{} extraction and math_verify
grading, and the parameters below are the accuracy command the cookbook
publishes for this model: 64 threads, 32768 max tokens, temperature 1.0,
top_p 0.95, thinking on. The seed pins the sampling so a failure is a
regression rather than a reroll.

Requires the ROCm enablement from #36607 (KPool/AITER paged-MQA, the non-gfx95
DSA top-k fallback, mHC gating, Quark mixed MXFP4/FP8 loading). Without it the
server does not reach a servable state on ROCm.

Registry: nightly-amd-8-gpu-mi35x-glm53-flash suite
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Register for AMD CI - MI35x GLM-5.3-Flash accuracy test (~90 min: the 328 GB
# checkpoint dominates startup, then full-split GSM8K with thinking at TP8)
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
            "--dsa-prefill-backend=tilelang",
            "--dsa-decode-backend=tilelang",
            "--kv-cache-dtype=bfloat16",
            "--moe-runner-backend=triton",
            "--reasoning-parser=glm45",
            "--tool-call-parser=glm47",
            # The cookbook cell's single `--disable-cuda-graph`; that legacy
            # flag is no longer on the CLI, so name both phases instead.
            "--disable-prefill-cuda-graph",
            "--disable-decode-cuda-graph",
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
                variant="TP8",
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
