"""MI30x GLM-5.3-Flash GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.3-Flash on MI30x (gfx942) with the ROCm recipe the cookbook
ships as `verified: true`: TP8, BF16 KV cache, TileLang DSA prefill+decode,
Triton MoE runner, AITER on, CUDA graphs off. Same command and same threshold
as the gfx950 gate in test_glm53_flash_eval_mi35x.py.

gfx942 is not redundant with gfx950 for this model. The AMD enablement routes
the two arches through different kernels for the same forward pass: gfx950 gets
the fused SGL-kernel k-pool top-k and the AITER mHC pre/post plus fused
attention-to-FFN boundary, while gfx942 takes the portable unfused Torch DSA
top-k and the generic mHC path. Those gfx942 paths have no other nightly
coverage. The cookbook's MI325X entry is inferred from this arch rather than
measured directly, so this job is the only evidence behind it too.

Threshold: #36607 measured the full 1319-question GSM8K split at
1284/1319 = 97.35% on MI300X. 0.92 follows this repo's `measured - 0.05`
convention for sgl-eval gsm8k thresholds and matches the gfx950 gate, so the
two arches stay directly comparable.

This harness scored 0.9712 on both the rocm720 image (12259 s wall) and the
rocm724 image (5164 s wall). The first budget of 7200 s timed out at 99.8%
complete because gfx942 has no AITER mHC and no fused DSA top-k -- both are
gfx95-gated -- so it evaluates roughly 3x slower than the MI35x job. The
workflow allows 18000 s.

Eval harness: `api="sgl_eval"` rather than the default 5-shot completion
scorer, because GLM-5.3-Flash thinks by default and the completion scorer reads
the last number in the response. The parameters below are the accuracy command
the cookbook publishes for this model. See the MI35x file for the longer note.

Model support is on main via #36507. The gfx942/gfx950 kernel split that this
job exists to gate is the AMD Day-0 stack (#36607 on the support branch;
reopened on main as #38541-#38547).

Registry: nightly-amd-accuracy-8-gpu-glm53-flash suite
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Register for AMD CI - MI30x GLM-5.3-Flash accuracy test. Measured 12259 s
# on rocm720 and 5164 s on rocm724; 13000 s covers the slower image plus
# a cold-cache load of the 328 GB checkpoint.
register_amd_ci(
    est_time=13000,
    suite="nightly-amd-accuracy-8-gpu-glm53-flash",
    nightly=True,
)

GLM_53_FLASH_MODEL_PATH = "zai-org/GLM-5.3-Flash"

# Fetching and loading a 328 GB checkpoint against a cold cache is what this
# budget has to cover; the default launch timeout is nowhere near enough.
SERVER_LAUNCH_TIMEOUT = 5400


class TestGLM53FlashEvalMI30x(unittest.TestCase):
    """GLM-5.3-Flash GSM8K Accuracy Evaluation Test for MI30x."""

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
            test_name="GLM-5.3-Flash (MI30x)",
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
