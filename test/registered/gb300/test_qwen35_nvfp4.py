import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300", nightly=True)

MODEL_PATH = "nvidia/Qwen3.5-397B-A17B-NVFP4"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=qwen3",
    "--tool-call-parser=qwen3_coder",
    "--quantization=modelopt_fp4",
    "--fp4-gemm-backend=flashinfer_cutlass",
    "--moe-runner-backend=flashinfer_trtllm",
    "--kv-cache-dtype=fp8_e4m3",
    "--enable-flashinfer-allreduce-fusion",
    "--attention-backend=trtllm_mha",
    "--mem-fraction-static=0.8",
    "--enable-multimodal",
    "--enable-metrics",
]

MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
    "--mamba-scheduler-strategy=extra_buffer",
    "--page-size=64",
]


class TestQwen35Nvfp4(unittest.TestCase):
    """Qwen3.5-397B NVFP4 on GB300 (4x B200 NVL4, tp=4)."""

    def test_qwen35_nvfp4(self):
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
                extra_args=COMMON_ARGS + ["--dp-size=4", "--enable-dp-attention"],
                variant="TP4+DP4+DPA",
            ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS
                + ["--dp-size=4", "--enable-dp-attention"]
                + MTP_ARGS,
                variant="TP4+DP4+DPA+MTP",
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-397B-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="mmmu-pro", baseline_accuracy=0.78, repeat=1, max_tokens=32768
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
