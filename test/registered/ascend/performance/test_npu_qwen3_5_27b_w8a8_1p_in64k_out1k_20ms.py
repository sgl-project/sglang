import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
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

QWEN3_5_27B_64K_1K_LOW_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_NPU_PROFILING": "1",
}

QWEN3_5_27B_64K_1K_LOW_OTHER_ARGS = [
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
    130000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    1,
    "--max-mamba-cache-size",
    10,
    "--mem-fraction-static",
    0.5,
    "--cuda-graph-bs",
    1,
    2,
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
    280000,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUQwen3_5_27B_1P_In64k_Out1k_Low(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.5-27B-W8A8 1p in64k out1k low latency"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_27B_W8A8_MODEL_PATH
    other_args = QWEN3_5_27B_64K_1K_LOW_OTHER_ARGS
    envs = QWEN3_5_27B_64K_1K_LOW_ENVS
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 4
    input_len = 65536
    output_len = 1024
    random_range_ratio = 1
    tpot = 20
    output_token_throughput = 48

    def test_npu_qwen3_5_27b_1p_in64k_out1k_low(self):
        """Run NPU performance test for Qwen3.5-27B-W8A8 in64k out1k low latency"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
