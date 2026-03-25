import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings
from sglang.test.tool_call_test_runner import ToolCallTestParams

register_cuda_ci(est_time=5400, suite="nightly-8-gpu-common", nightly=True)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

BASE_ARGS = [
    "--trust-remote-code",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]

TOOL_CALL_ARGS = [
    "--tool-call-parser=deepseekv32",
    "--reasoning-parser=deepseek-v3",
]

DP_ARGS = [
    "--tp=8",
    "--dp=8",
    "--enable-dp-attention",
]

# Accuracy thresholds
GSM8K_BASELINE = 0.935
GPQA_BASELINE = 0.83


class TestDeepseekV32IndexCache(unittest.TestCase):
    """Unified test class for DeepSeek V3.2 performance and accuracy with IndexCache enabled.

    Tests multiple variants with both performance and accuracy tests:
    - dp: Standard TP=8 + DP=8 with dp-attention + IndexCache
    - tp: Pure TP=8 only + IndexCache
    """

    def test_deepseek_v32_uniform_interleaving(self):
        """Run performance and accuracy for all DeepSeek V3.2 variants."""
        TP_ARGS = [
            "--tp=8",
        ]
        INDEXCACHE_FREQ_ARGS = [
            "--json-model-override-args",
            '{"index_topk_freq": 4}',
        ]
        INDEXCACHE_PATTERN_ARGS = [
            "--json-model-override-args",
            '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSF"}',
        ]
        variants = [
            # Variant: "dp" - Standard TP=8 + DP=8 with dp-attention + IndexCache with uniform frequency of 4
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + TOOL_CALL_ARGS + INDEXCACHE_FREQ_ARGS,
                variant="DP8+IndexCacheFreq",
            ),
            # Variant: "dp" - Standard TP=8 + DP=8 with dp-attention + IndexCache with custom pattern
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS
                + DP_ARGS
                + TOOL_CALL_ARGS
                + INDEXCACHE_PATTERN_ARGS,
                variant="DP8+IndexCachePattern",
            ),
            # Variant: "tp" - Pure TP=8 only + IndexCache with uniform frequency of 4
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + TP_ARGS + TOOL_CALL_ARGS + INDEXCACHE_FREQ_ARGS,
                variant="TP8+IndexCacheFreq",
            ),
            # Variant: "tp" - Pure TP=8 only + IndexCache with custom pattern
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS
                + TP_ARGS
                + TOOL_CALL_ARGS
                + INDEXCACHE_PATTERN_ARGS,
                variant="TP8+IndexCachePattern",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=GSM8K_BASELINE
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 8, 16, 64],
                profile_dir="performance_profiles_deepseek_v32",
            ),
            tool_call_params=ToolCallTestParams(
                test_thinking=True, test_reasoning_usage=True
            ),
        )


if __name__ == "__main__":
    unittest.main()
