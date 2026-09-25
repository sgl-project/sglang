"""DIAGNOSTIC ONLY -- not for merge.

Scratch variant of the MI30x GLM-5.3-Flash gate, used to localize why gfx942
generates non-terminating output on main while gfx950 scores 0.975 on the same
recipe. Run 36092686822 got the server up and the decode graphs captured (the
two HIP fixes in #41136 work), then ran ~21.8k generated tokens per sequence
without stopping and was killed by the 18000 s per-file budget after finishing
roughly 106 of 1319 questions. gfx950 on the same eval peaked at ~398 tokens
per sequence.

Three suspects differ between the two arches; this file runs one small eval per
suspect in a single job so the 328 GB weight load is paid once and the page
cache carries the later launches:

  1. baseline      -- current recipe: fused sgl-kernel DSA top-k, Triton MoE,
                      unfused torch mHC. Expected to reproduce the runaway.
  2. torch-topk    -- #36607's non-gfx95 ROCm configuration: portable Torch DSA
                      top-k with the fused top-k paths off. Isolates the fused
                      sgl-kernel top-k, which no gfx942 run has ever validated.
  3. aiter-moe     -- AITER MoE runner instead of Triton, matching what gfx950
                      uses. Isolates the Triton MoE runner.

If 2 is healthy the #36607 top-k fallback is the fix and belongs in #41136. If 3
is healthy the Triton MoE runner is at fault. If neither is healthy, suspicion
falls on the unfused torch mHC path that #41136 now routes gfx942 to.

64 questions and 2048 max tokens keep each variant short: gfx950 needed under
400 tokens per answer, and a runaway variant costs at most ~8 minutes at the
266 token/s this arch sustains. The threshold is 0.0 so every variant runs and
reports its score instead of the first failure ending the job.
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_amd_ci(
    est_time=13000,
    suite="nightly-amd-accuracy-8-gpu-glm53-flash",
    nightly=True,
)

GLM_53_FLASH_MODEL_PATH = "zai-org/GLM-5.3-Flash"

SERVER_LAUNCH_TIMEOUT = 5400

COOKBOOK_ARGS = [
    "--ep-size=8",
    "--attention-backend=dsa",
    "--dsa-prefill-backend=tilelang",
    "--dsa-decode-backend=tilelang",
    "--linear-attn-backend=triton",
    "--kv-cache-dtype=bfloat16",
    "--cuda-graph-backend-decode=full",
    "--cuda-graph-backend-prefill=disabled",
    "--cuda-graph-bs-decode",
    "1",
    "32",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
    "--watchdog-timeout=1200",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]


class TestGLM53FlashEvalMI30x(unittest.TestCase):
    """GLM-5.3-Flash gfx942 output-quality localization."""

    def test_glm_53_flash(self):
        variants = [
            ModelLaunchSettings(
                GLM_53_FLASH_MODEL_PATH,
                tp_size=8,
                extra_args=COOKBOOK_ARGS + ["--moe-runner-backend=triton"],
                env={"SGLANG_USE_AITER": "1"},
                variant="baseline-fused-topk-triton-moe",
                launch_timeout=SERVER_LAUNCH_TIMEOUT,
            ),
            ModelLaunchSettings(
                GLM_53_FLASH_MODEL_PATH,
                tp_size=8,
                extra_args=COOKBOOK_ARGS
                + ["--moe-runner-backend=triton", "--dsa-topk-backend=torch"],
                env={
                    "SGLANG_USE_AITER": "1",
                    "SGLANG_DSA_FUSE_TOPK": "0",
                    "SGLANG_OPT_USE_TOPK_V2": "0",
                },
                variant="torch-topk-triton-moe",
                launch_timeout=SERVER_LAUNCH_TIMEOUT,
            ),
            ModelLaunchSettings(
                GLM_53_FLASH_MODEL_PATH,
                tp_size=8,
                extra_args=COOKBOOK_ARGS + ["--moe-runner-backend=aiter"],
                env={"SGLANG_USE_AITER": "1"},
                variant="fused-topk-aiter-moe",
                launch_timeout=SERVER_LAUNCH_TIMEOUT,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-5.3-Flash gfx942 localization (MI30x)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.0,
                api="sgl_eval",
                num_examples=64,
                num_threads=64,
                max_tokens=2048,
                temperature=1.0,
                top_p=0.95,
                seed=42,
                sgl_eval_thinking=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
