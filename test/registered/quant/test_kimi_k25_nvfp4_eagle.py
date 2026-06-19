import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Kimi-K2.5 NVFP4 + EAGLE3 (MLA draft) speculative decoding on 4x B200, tp=4.
register_cuda_ci(est_time=3000, suite="nightly-4-gpu-b200", nightly=True)

MODEL_PATH = "nvidia/Kimi-K2.5-NVFP4"
DRAFT_MODEL_PATH = "lightseekorg/kimi-k2.5-eagle3-mla"

EXTRA_ARGS = [
    "--trust-remote-code",
    "--attention-backend=tokenspeed_mla",
    "--moe-runner-backend=flashinfer_trtllm",
    "--quantization=modelopt_fp4",
    "--kv-cache-dtype=fp8_e4m3",
    "--mem-fraction-static=0.85",
    "--max-running-requests=16",
    "--speculative-algorithm=EAGLE3",
    f"--speculative-draft-model-path={DRAFT_MODEL_PATH}",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
    "--speculative-draft-model-quantization=unquant",
]


class TestKimiK25Nvfp4Eagle(unittest.TestCase):
    """Kimi-K2.5 NVFP4 with EAGLE3 speculative decoding on 4x B200 (tp=4).

    Runs both an accuracy test (gsm8k) and a performance test (bs=1/8/16),
    and gates the speculative-decoding accept length.
    """

    def test_kimi_k25_nvfp4_eagle(self):
        variants = [
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=EXTRA_ARGS,
                variant="TP4+EAGLE3",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.5-NVFP4 EAGLE3",
            # Thresholds from a measured tp=4 run: gsm8k 0.945, perf accept ~3.0-3.3.
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.92,
                num_examples=200,
                api="completion",
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 8, 16],
                spec_accept_length_threshold=2.8,
                profile_dir="performance_profiles_kimi_k25_nvfp4_eagle",
            ),
        )


if __name__ == "__main__":
    unittest.main()
