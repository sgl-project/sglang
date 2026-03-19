import os
import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

# Runs on both H200 and B200 via nightly-8-gpu-common suite
# Note: trtllm_mla backend may have hardware-specific behavior
register_cuda_ci(est_time=3000, suite="nightly-8-gpu-common", nightly=True)

MISTRAL_LARGE3_FP8_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"
MISTRAL_LARGE3_NVFP4_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4"
MISTRAL_LARGE3_EAGLE_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle"


@unittest.skipIf(not is_blackwell_system(), "Requires B200")
class TestMistralLarge3(unittest.TestCase):
    """Unified test class for Mistral-Large-3 performance and accuracy.

    Three variants:
    - basic: FP8 model + TP=8 + trtllm_mla backend
    - eagle: basic + EAGLE speculative decoding with draft model
    - nvfp4: NVFP4 model + TP=8 + trtllm_mla backend

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    @classmethod
    def setUpClass(cls):
        # Set environment variable to disable JIT DeepGemm
        os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

    @classmethod
    def tearDownClass(cls):
        # Clean up environment variable
        if "SGLANG_ENABLE_JIT_DEEPGEMM" in os.environ:
            del os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"]

    def test_mistral_large3_all_variants(self):
        """Run performance and accuracy for all Mistral-Large-3 variants."""
        base_args = [
            "--tp=8",
            "--attention-backend=trtllm_mla",
            "--moe-runner-backend=flashinfer_trtllm",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--chat-template=mistral",
        ]
        eagle_args = [
            "--speculative-algorithm=EAGLE",
            f"--speculative-draft-model-path={MISTRAL_LARGE3_EAGLE_MODEL_PATH}",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
            "--kv-cache-dtype=auto",
        ]

        variants = [
            # Variant: "basic" - FP8 model + TP=8 + trtllm_mla backend
            ModelLaunchSettings(
                MISTRAL_LARGE3_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
            # Variant: "eagle" - FP8 model + TP=8 + trtllm_mla + EAGLE with draft model
            ModelLaunchSettings(
                MISTRAL_LARGE3_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + eagle_args,
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
                variant="TP8+MTP",
            ),
            # Variant: "nvfp4" - NVFP4 model + TP=8 + trtllm_mla backend
            ModelLaunchSettings(
                MISTRAL_LARGE3_NVFP4_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="NVFP4",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Mistral-Large-3",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.85),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_mistral_large3",
            ),
        )


if __name__ == "__main__":
    unittest.main()
