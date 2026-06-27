import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(
    est_time=7200, suite="nightly-4-gpu-gb300-deepseek-v4-pro-fp4", nightly=True
)

MODEL_PATH = "deepseek-ai/DeepSeek-V4-Pro"
SERVER_LAUNCH_TIMEOUT = 3600

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

LOW_LATENCY_ARGS = [
    "--trust-remote-code",
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--chunked-prefill-size",
    "8192",
    "--disable-flashinfer-autotune",
    "--swa-full-tokens-ratio",
    "0.1",
    "--mem-fraction-static",
    "0.85",
]

BALANCED_ARGS = [
    "--trust-remote-code",
    "--dp",
    "4",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
    "--mem-fraction-static",
    "0.85",
    "--cuda-graph-max-bs",
    "128",
    "--max-running-requests",
    "256",
    "--deepep-config",
    DEEPEP_CONFIG,
]

HIGH_THROUGHPUT_ARGS = [
    "--trust-remote-code",
    "--dp",
    "4",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "megamoe",
    "--mem-fraction-static",
    "0.9",
    "--cuda-graph-max-bs",
    "128",
    "--max-running-requests",
    "256",
]

BALANCED_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}

HIGH_THROUGHPUT_ENV = {
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
}

PERFORMANCE_BATCH_SIZES = {
    "low-latency": [1, 4, 16],
    "balanced": [64],
    "high-throughput": [128],
}


class TestDeepSeekV4ProFp4(unittest.TestCase):
    """DeepSeek-V4-Pro FP4 on GB300 (4x B200 NVL4, tp=4)."""

    def test_deepseek_v4_pro_fp4(self):
        variants = [
            # ModelLaunchSettings(
            #     MODEL_PATH,
            #     tp_size=4,
            #     extra_args=LOW_LATENCY_ARGS,
            #     variant="low-latency",
            #     launch_timeout=SERVER_LAUNCH_TIMEOUT,
            # ),
            # ModelLaunchSettings(
            #     MODEL_PATH,
            #     tp_size=4,
            #     extra_args=BALANCED_ARGS,
            #     env=BALANCED_ENV,
            #     variant="balanced",
            #     launch_timeout=SERVER_LAUNCH_TIMEOUT,
            # ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=HIGH_THROUGHPUT_ARGS,
                env=HIGH_THROUGHPUT_ENV,
                variant="high-throughput",
                launch_timeout=SERVER_LAUNCH_TIMEOUT,
            ),
        ]

        failures = []
        accuracy_params = AccuracyTestParams(
            dataset="gsm8k",
            baseline_accuracy=0.935,
            temperature=1.0,
            top_p=1.0,
        )
        for variant in variants:
            try:
                run_combined_tests(
                    models=[variant],
                    test_name=f"DeepSeek-V4-Pro-FP4 ({variant.variant})",
                    accuracy_params=accuracy_params,
                    performance_params=PerformanceTestParams(
                        batch_sizes=PERFORMANCE_BATCH_SIZES[variant.variant],
                        profile_dir="performance_profiles_gb300",
                    ),
                )
            except AssertionError as e:
                failures.append(f"{variant.variant}: {e}")

        if failures:
            raise AssertionError(
                "DeepSeek-V4-Pro-FP4 failures:\n" + "\n".join(failures)
            )


if __name__ == "__main__":
    unittest.main()
