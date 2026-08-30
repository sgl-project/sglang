import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gb300_utils import GB300_NCCL_PORT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=7200, stage="nightly", runner_config="4-gpu-gb300")

MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=qwen3",
    "--tool-call-parser=qwen3_coder",
    "--enable-flashinfer-allreduce-fusion",
    "--attention-backend=trtllm_mha",
    "--mem-fraction-static=0.8",
    "--mamba-scheduler-strategy=extra_buffer",
    "--enable-multimodal",
    "--enable-metrics",
    "--nccl-port",
    GB300_NCCL_PORT,
]

DP_MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=1",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=2",
]


class TestQwen35Fp8Dp(CustomTestCase):
    """Qwen3.5-397B FP8 DP4+DPA+MTP on GB300 (4x GB300 NVL4)."""

    def test_qwen35_fp8_dp(self):
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
                    + DP_MTP_ARGS,
                    variant="TP4+DP4+DPA+MTP",
                )
            ],
            test_name="Qwen3.5-397B-FP8 (TP4+DP4+DPA+MTP)",
            accuracy_params=AccuracyTestParams(
                dataset="mmmu_pro_vision",
                baseline_accuracy=0.76,
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
