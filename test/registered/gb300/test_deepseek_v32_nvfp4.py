import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300", nightly=True)

MODEL_PATH = "nvidia/DeepSeek-V3.2-NVFP4"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=deepseek-v3",
    "--tool-call-parser=deepseekv32",
    "--quantization=modelopt_fp4",
    "--moe-runner-backend=flashinfer_trtllm",
    "--kv-cache-dtype=bfloat16",
    "--mem-fraction-static=0.8",
    "--enable-metrics",
]

MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
]


class TestDeepseekV32Nvfp4(unittest.TestCase):
    """DeepSeek V3.2 NVFP4 on GB300 (4x B200 NVL4, tp=4)."""

    def test_deepseek_v32_nvfp4(self):
        variants = [
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS,
                variant="TP4",
            ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS
                + [
                    "--dp-size=4",
                    "--ep-size=4",
                    "--enable-dp-attention",
                ],
                variant="TP4+DP4+DPA",
            ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS
                + [
                    "--dp-size=4",
                    "--ep-size=4",
                    "--enable-dp-attention",
                ]
                + MTP_ARGS,
                variant="TP4+DP4+DPA+MTP",
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=0.935
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
