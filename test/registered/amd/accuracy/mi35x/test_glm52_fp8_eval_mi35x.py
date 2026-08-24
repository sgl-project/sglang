"""MI35x GLM-5.2-FP8 GSM8K Accuracy Evaluation Test (8-GPU)

Tests zai-org/GLM-5.2-FP8 with the DSA tilelang backend on MI35x (gfx950).

Server arguments track the GLM-5.2 cookbook's MI355X / FP8 / low-latency /
single-node cell: TP8, DSA tilelang prefill+decode, 131072 chunked prefill,
0.80 static memory fraction, and a 20-minute watchdog for weight loading.
That cell ships as `verified: true` with published benchmarks, but nothing in
CI re-checks it, so this nightly is what keeps it honest.

gfx950 is the arch that needs the guard. An earlier ROCm 7.2 miscompile of
aiter's block-FP8 `gemm_a8w8_blockscale_bpreshuffle` GEMM was small per layer
but compounded across all 78 layers: GSM8K collapsed to ~0 while short factual
prompts still looked fine. Only a multi-step reasoning eval catches that class
of regression, and MI300X/MI325X (gfx942) were never affected -- hence MI35x
only.

ROCm 7.2 only, and that is a measured decision rather than a cautious one. Both
lanes were run on current same-day images (`v0.5.16-rocm7{00,20}-mi35x-20260727`)
and 7.2 scored 0.931 while 7.0 scored 0.270. The 7.0 image ships Triton 3.4.0,
which is below the >=3.5.0 the aiter gluon DSA kernels need, so it logs
"falling back to legacy page_size=1 / KVBlockSize=1 path" and that fallback is
what loses the accuracy. Re-add a 7.0 job once its image ships Triton >=3.5.0,
or once the legacy DSA fallback is fixed on gfx950.

The eval matches the CUDA GLM-5.2-FP8 nightly (`test/registered/8-gpu-models/
test_glm52_fp8.py`): same dataset and same 0.92 baseline, so a red run here
means AMD diverged from CUDA rather than the harness diverging.

Headroom is thinner than the cookbook's ~0.96 suggests: this harness measured
0.931 against a 0.92 baseline. The cookbook number comes from a different GSM8K
setup, so the two are not directly comparable, but it does mean a ~1 point
regression trips this test. Treat a marginal failure as worth reading rather
than as noise to retry away.

Accuracy runs before the companion performance benchmark in the same workflow
job. A correctness failure therefore skips performance, while a green result
lets the benchmark reuse the cached checkpoint and the same scarce runner slot.

Registry: nightly-amd-8-gpu-mi35x-glm52-fp8 suite
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Register for AMD CI - MI35x GLM-5.2-FP8 accuracy test (~90 min: the ~700 GB
# FP8 checkpoint dominates startup, then GSM8K 5-shot at TP8)
register_amd_ci(
    est_time=5400,
    suite="nightly-amd-8-gpu-mi35x-glm52-fp8",
    nightly=True,
)

GLM_52_FP8_MODEL_PATH = "zai-org/GLM-5.2-FP8"

# Fetching and loading a ~700 GB checkpoint against a cold cache is what this
# budget has to cover; the default launch timeout is nowhere near enough.
SERVER_LAUNCH_TIMEOUT = 5400


class TestGLM52FP8EvalMI35x(unittest.TestCase):
    """GLM-5.2-FP8 GSM8K Accuracy Evaluation Test for MI35x."""

    def test_glm_52_fp8(self):
        """Run accuracy test for GLM-5.2-FP8."""
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

        variants = [
            ModelLaunchSettings(
                GLM_52_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=cookbook_args,
                variant="TP8",
                launch_timeout=SERVER_LAUNCH_TIMEOUT,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-5.2-FP8 (MI35x)",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.92),
        )


if __name__ == "__main__":
    unittest.main()
