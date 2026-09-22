import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gb300_utils import GB300_NCCL_PORT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=7200, stage="nightly", runner_config="4-gpu-gb300")

MODEL_PATH = "nvidia/Kimi-K2.5-NVFP4"
DRAFT_MODEL_PATH = "lightseekorg/kimi-k2.5-eagle3-mla"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=kimi_k2",
    "--tool-call-parser=kimi_k2",
    "--quantization=modelopt_fp4",
    "--attention-backend=tokenspeed_mla",
    "--kv-cache-dtype=fp8_e4m3",
    "--moe-runner-backend=flashinfer_trtllm",
    "--mem-fraction-static=0.8",
    "--enable-metrics",
    "--speculative-algorithm=EAGLE3",
    f"--speculative-draft-model-path={DRAFT_MODEL_PATH}",
    "--speculative-draft-model-quantization=unquant",
    "--nccl-port",
    GB300_NCCL_PORT,
]

DP_EAGLE_ARGS = [
    "--speculative-num-steps=1",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=2",
]


class TestKimiK25Nvfp4Dp(CustomTestCase):
    """Kimi-K2.5 NVFP4 DP4+DPA+EAGLE3 on GB300 (4x GB300 NVL4)."""

    def test_kimi_k25_nvfp4_dp(self):
        # Pinned to what `ns eval --benchmarks=mmmu-pro:1` sent implicitly --
        # its `:1` suffix means temperature 0.7, not greedy -- so the baseline
        # carries over unchanged. Do not "simplify" these away.
        run_combined_tests(
            models=[
                ModelLaunchSettings(
                    MODEL_PATH,
                    tp_size=4,
                    extra_args=COMMON_ARGS
                    + ["--dp-size=4", "--enable-dp-attention"]
                    + DP_EAGLE_ARGS,
                    variant="TP4+DP4+DPA+EAGLE3",
                )
            ],
            test_name="Kimi-K2.5-NVFP4 (TP4+DP4+DPA+EAGLE3)",
            accuracy_params=AccuracyTestParams(
                dataset="mmmu_pro_vision",
                baseline_accuracy=0.69,
                repeat=1,
                max_tokens=32768,
                temperature=0.7,
                seed=0,
                sgl_eval_thinking=False,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                result_dir="performance_results_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
