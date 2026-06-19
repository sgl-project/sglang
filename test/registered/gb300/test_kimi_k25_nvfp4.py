import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300", nightly=True)

MODEL_PATH = "nvidia/Kimi-K2.5-NVFP4"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=kimi_k2",
    "--tool-call-parser=kimi_k2",
    "--quantization=modelopt_fp4",
    "--attention-backend=tokenspeed_mla",
    "--kv-cache-dtype=fp8_e4m3",
    "--moe-runner-backend=flashinfer_trtllm",
    "--mem-fraction-static=0.75",
    "--enable-multimodal",
    "--enable-metrics",
]

TP_MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
]

DP_MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=1",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=2",
]


class TestKimiK25Nvfp4(unittest.TestCase):
    """Kimi-K2.5 NVFP4 on GB300 (4x GB300 NVL4, tp=4).

    No EAGLE/MTP support for Kimi-K2.5 — only TP and TP+DP+DPA variants.
    """

    def test_kimi_k25_nvfp4(self):
        variants = [
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS + TP_MTP_ARGS,
                variant="TP4+MTP",
            ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS
                + ["--dp-size=4", "--enable-dp-attention"]
                + DP_MTP_ARGS,
                variant="TP4+DP4+DPA+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.5-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="mmmu-pro", baseline_accuracy=0.69, repeat=1, max_tokens=32768
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
