import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_27B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_5_27B_W8A8_IN1024X1024_30_OUT1024_50MS_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "ASCEND_USE_FIA": "1",
}

QWEN3_5_27B_W8A8_IN1024X1024_30_OUT1024_50MS_OTHER_ARGS = [
    "--model-path",
    QWEN3_5_27B_W8A8_MODEL_PATH,
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    81920,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    48,
    "--max-mamba-cache-size",
    80,
    "--mem-fraction-static",
    0.6,
    "--cuda-graph-bs",
    2,
    8,
    16,
    24,
    36,
    42,
    48,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--max-total-tokens",
    850000,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUQwen3_5_27B_W8A8_1P_In1024x1024_30_Out1024_50ms(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for Qwen3.5-27B-W8A8 1p in1024x1024 30 out1024 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    model = QWEN3_5_27B_W8A8_MODEL_PATH
    other_args = QWEN3_5_27B_W8A8_IN1024X1024_30_OUT1024_50MS_OTHER_ARGS
    envs = QWEN3_5_27B_W8A8_IN1024X1024_30_OUT1024_50MS_ENVS
    dataset_name = "image"
    image_resolution = "1024x1024"
    image_count = 1
    max_concurrency = 16
    num_prompts = 16
    input_len = 30
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 30

    def test_npu_qwen3_5_27b_w8a8_1p_in1024x1024_30_out1024_50ms(self):
        """Run NPU performance test for Qwen3.5-27B-W8A8 1p in1024x1024 30 out1024 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
