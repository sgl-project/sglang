"""MI30x GLM-5.3-Flash GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.3-Flash on MI30x (gfx942) with the AMD FP8 recipe from the
GLM-5.3-Flash cookbook refresh (#36712): TP8 + EP8, BF16 KV cache, TileLang DSA
prefill+decode, Triton linear attention, Triton MoE runner, SGLANG_USE_AITER=1,
full decode graphs at batch sizes 1 and 32. Same eval and threshold as the
gfx950 gate in test_glm53_flash_eval_mi35x.py.

gfx942 is not redundant with gfx950 for this model. It takes the Triton MoE
runner and the generic mHC path (AITER mHC is gfx95-only), and neither has any
other nightly coverage for this model. It also resolves `dsa_topk_backend` to
the default fused `sgl-kernel` top-k on main, whereas the support-branch runs
behind #36607's numbers forced the portable Torch top-k on non-gfx95 ROCm.

Threshold: #36607 measured the full 1319-question GSM8K split at
1284/1319 = 97.35% on MI300X, and the cookbook's MI300X cell for this exact
TP8 + EP8 command measured 1280/1319 = 97.04%, both on the GLM-5.3-Flash
support branch. 0.92 follows this repo's `measured - 0.05` convention for
sgl-eval gsm8k thresholds and matches the gfx950 gate, so the two arches stay
directly comparable.

Runtime: budget this job generously. gfx942 gets none of the gfx95 fast paths,
and this runner pool's shared model cache has taken over an hour to load the
328 GB checkpoint. The workflow allows 18000 s. If that ever proves tight,
prefer raising it over trimming the eval: a full-split score is what makes this
arch's number comparable to the gfx950 one.

Eval harness: `api="sgl_eval"` rather than the default 5-shot completion
scorer, because GLM-5.3-Flash thinks by default and the completion scorer reads
the last number in the response. The parameters below are the accuracy command
the cookbook publishes for this model. See the MI35x file for the longer note.

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
            "--ep-size=8",
            "--attention-backend=dsa",
            "--dsa-prefill-backend=tilelang",
            "--dsa-decode-backend=tilelang",
            "--linear-attn-backend=triton",
            "--kv-cache-dtype=bfloat16",
            "--moe-runner-backend=triton",
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
