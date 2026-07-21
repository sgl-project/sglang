import unittest

from sglang.srt.environ import envs
from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

# NVFP4 needs Blackwell FP4 kernels, so this runs on the Blackwell leg of the
# common 8-GPU suite (Hopper is skipped below).
register_cuda_ci(est_time=3600, suite="nightly-8-gpu-common", nightly=True)

INKLING_NVFP4_MODEL = "thinkingmachines/Inkling-NVFP4"

# Verified Blackwell NVFP4 recipe; see the SGLang cookbook Inkling page.
NVFP4_ARGS = [
    "--tp=8",
    "--trust-remote-code",
    "--quantization=modelopt_fp4",
    "--attention-backend=fa4",
    "--page-size=128",
    "--fp4-gemm-backend=flashinfer_trtllm",
    "--moe-runner-backend=flashinfer_trtllm_routed",
    "--enable-torch-symm-mem",
    "--mamba-radix-cache-strategy=extra_buffer",
    "--mem-fraction-static=0.85",
    "--swa-full-tokens-ratio=0.1",
    "--mamba-full-memory-ratio=0.1",
    "--reasoning-parser=inkling",
]

# gsm8k measured ~0.98 (temp 0, 300 examples); floored with margin for the
# temperature=1.0 nightly config and FP4-kernel variance.
GSM8K_BASELINE = 0.93


class TestInklingNVFP4Nightly(unittest.TestCase):
    """Nightly test for Inkling-NVFP4 (975B MoE), TP=8, Blackwell only.

    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    @unittest.skipIf(not is_blackwell_system(), "NVFP4 requires Blackwell")
    def test_inkling_nvfp4(self):
        """Run performance and accuracy for Inkling-NVFP4 (TP8)."""
        variants = [
            ModelLaunchSettings(
                INKLING_NVFP4_MODEL,
                tp_size=8,
                extra_args=NVFP4_ARGS,
                variant="TP8",
            ),
        ]

        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(1):
            run_combined_tests(
                models=variants,
                test_name="Inkling-NVFP4",
                accuracy_params=AccuracyTestParams(
                    dataset="gsm8k",
                    baseline_accuracy=GSM8K_BASELINE,
                    num_examples=1314,
                    num_threads=512,
                    max_tokens=16000,
                    temperature=1.0,
                    top_p=0.95,
                    repeat=1,
                ),
                performance_params=PerformanceTestParams(
                    profile_dir="performance_profiles_inkling_nvfp4",
                ),
            )


if __name__ == "__main__":
    unittest.main()
