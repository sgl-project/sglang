import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_27B_W8A8_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_27B_16K_1k_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "GDN_ATTN_BACKEND_TRITON": "1",
    "ASCEND_USE_FIA": "1",
}

QWEN3_6_27B_16K_1k_OTHER_ARGS = [
    "--tp-size",
    4,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    58000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    29,
    "--max-mamba-cache-size",
    58,
    "--mem-fraction-static",
    0.68,
    "--cuda-graph-bs",
    1,
    2,
    8,
    12,
    16,
    20,
    24,
    26,
    28,
    29,
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]


class TestNPUQwen3_6_27B_2P_In16k_Out1k_50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.6-27B-w8a8 2p in16k out1k 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    other_args = QWEN3_6_27B_16K_1k_OTHER_ARGS
    envs = QWEN3_6_27B_16K_1k_ENVS
    dataset_name = "random"
    max_concurrency = 29
    num_prompts = 116
    input_len = 16000
    output_len = 1000
    random_range_ratio = 1
    seed = 1
    tpot = 50
    output_token_throughput = 426.1

    def test_npu_qwen3_6_27b_2p_in16k_out1k_50ms(self):
        """Run NPU performance test for Qwen3.6-27B-w8a8 in16k out1k 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
