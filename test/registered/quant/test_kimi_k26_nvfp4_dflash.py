import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Kimi-K2.6 NVFP4 (pure-MLA target, fp8 KV) + DFlash speculative decoding on 8x B200, tp=8.
register_cuda_ci(est_time=3600, suite="nightly-8-gpu-b200", nightly=True)

MODEL_PATH = "nvidia/Kimi-K2.6-NVFP4"
DRAFT_MODEL_PATH = "nvidia/Kimi-K2.6-DFlash"

# trtllm_mla verify only; cuteDSL fold verify depends on the flashinfer version.
EXTRA_ARGS = [
    "--trust-remote-code",
    "--quantization=modelopt_fp4",
    "--moe-runner-backend=flashinfer_trtllm",
    "--fp4-gemm-backend=flashinfer_cutlass",
    "--attention-backend=trtllm_mla",
    "--kv-cache-dtype=fp8_e4m3",
    "--mem-fraction-static=0.85",
    "--max-running-requests=16",
    "--speculative-algorithm=DFLASH",
    f"--speculative-draft-model-path={DRAFT_MODEL_PATH}",
    "--speculative-num-draft-tokens=8",
    "--speculative-draft-attention-backend=fa4",
    "--speculative-draft-model-quantization=unquant",
    "--speculative-draft-window-size=4096",
]


class TestKimiK26Nvfp4Dflash(unittest.TestCase):
    """Kimi-K2.6 NVFP4 (pure-MLA, fp8 KV) with DFlash speculative decoding on 8x B200 (tp=8).

    Runs both an accuracy test (gsm8k) and a performance test (bs=1/8/16), and gates the
    speculative-decoding accept length. Guards the pure-MLA fp8-KV DFlash path.
    """

    def test_kimi_k26_nvfp4_dflash(self):
        variants = [
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=8,
                extra_args=EXTRA_ARGS,
                variant="TP8+DFLASH",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.6-NVFP4 DFlash",
            # Thresholds from a measured tp=8 run: gsm8k 0.936 (full set), accept length ~2.66.
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.92,
                num_examples=200,
                api="completion",
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 8, 16],
                spec_accept_length_threshold=2.0,
                profile_dir="performance_profiles_kimi_k26_nvfp4_dflash",
            ),
        )


if __name__ == "__main__":
    unittest.main()
