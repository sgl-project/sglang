import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
# Higher est_time due to 6 variants with both performance and accuracy tests
register_cuda_ci(est_time=3600, suite="nightly-8-gpu-common", nightly=True)

GPT_OSS_20B_BF16_MODEL_PATH = "lmsys/gpt-oss-20b-bf16"
GPT_OSS_20B_MXFP4_MODEL_PATH = "openai/gpt-oss-20b"
GPT_OSS_20B_EAGLE3_DRAFT_MODEL_PATH = "zhuyksir/EAGLE3-gpt-oss-20b-bf16"


class TestGptOss20B(unittest.TestCase):
    """Unified test class for GPT-OSS-20B performance and accuracy.

    Six variants testing combinations of:
    - Quantization: BF16 vs MXFP4
    - Parsers: With/without reasoning-parser and tool-call-parser
    - Speculative Decoding: With/without EAGLE3

    Variants:
    1. BF16 baseline
    2. MXFP4 baseline
    3. BF16 + Parsers
    4. MXFP4 + Parsers
    5. BF16 + Parsers + EAGLE3 (full featured)
    6. MXFP4 + Parsers + EAGLE3 (full featured quantized)

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    def test_gpt_oss_20b_all_variants(self):
        """Run performance and accuracy for all GPT-OSS-20B variants."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
            "--cuda-graph-max-bs=600",
        ]
        parser_args = [
            "--reasoning-parser=gpt-oss",
            "--tool-call-parser=gpt-oss",
        ]
        eagle3_args = [
            "--speculative-algorithm=EAGLE3",
            f"--speculative-draft-model-path={GPT_OSS_20B_EAGLE3_DRAFT_MODEL_PATH}",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ]
        eagle3_env = {
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }

        variants = [
            # Variant 1: BF16 baseline
            ModelLaunchSettings(
                GPT_OSS_20B_BF16_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="BF16",
            ),
            # Variant 2: MXFP4 baseline
            ModelLaunchSettings(
                GPT_OSS_20B_MXFP4_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="MXFP4",
            ),
            # Variant 3: BF16 + Parsers
            ModelLaunchSettings(
                GPT_OSS_20B_BF16_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + parser_args,
                variant="BF16+Parsers",
            ),
            # Variant 4: MXFP4 + Parsers
            ModelLaunchSettings(
                GPT_OSS_20B_MXFP4_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + parser_args,
                variant="MXFP4+Parsers",
            ),
            # Variant 5: BF16 + Parsers + EAGLE3 (full featured)
            ModelLaunchSettings(
                GPT_OSS_20B_BF16_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + parser_args + eagle3_args,
                env=eagle3_env,
                variant="BF16+Parsers+EAGLE3",
            ),
            # Variant 6: MXFP4 + Parsers + EAGLE3 (full featured quantized)
            ModelLaunchSettings(
                GPT_OSS_20B_MXFP4_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + parser_args + eagle3_args,
                env=eagle3_env,
                variant="MXFP4+Parsers+EAGLE3",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GPT-OSS-20B",
            accuracy_params=None,
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_gpt_oss_20b",
            ),
        )


if __name__ == "__main__":
    unittest.main()
