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
per sequence, and the AITER MoE runner scored 63/64 = 0.984 at 151 tokens per
sequence. The DSA top-k backend makes no difference, so the fault is the Triton
MoE runner itself, and the cookbook's MI300X MoE recommendation is stale.

gfx942 is not redundant with gfx950 for this model. It runs the generic mHC
path, since AITER mHC is gfx95-only, and nothing else gives that path nightly
coverage for this model. That path reaches gfx942 only with the HIP guard in
#41136: without it, TileLang's HIP codegen cannot lower the tl.get_lane_idx in
the fused mHC post/pre kernel and decode graph capture dies with "Unresolved
call Op(tl.get_lane_idx)" (run 36092686822).

Threshold: 0.92 follows this repo's `measured - 0.05` convention for sgl-eval
gsm8k thresholds and matches the gfx950 gate, so the two arches stay directly
comparable. The 0.984 above is a 64-question sample; gfx950 scored 0.9750
(1286/1319) on the full split with the same eval.

Runtime: the 328 GB checkpoint has taken 3606-4143 s to load from this pool's
shared cache, and a full-split eval at the AITER MoE throughput measured above
adds roughly 900 s. The workflow allows 18000 s. If that ever proves tight,
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

# Register for AMD CI - MI30x GLM-5.3-Flash accuracy test. A cold-cache load
# of the 328 GB checkpoint has measured up to 4143 s and the full-split eval
# adds roughly 900 s on the AITER MoE runner; 9000 s leaves room for a slower
# image on top of that.
register_amd_ci(
    est_time=9000,
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
