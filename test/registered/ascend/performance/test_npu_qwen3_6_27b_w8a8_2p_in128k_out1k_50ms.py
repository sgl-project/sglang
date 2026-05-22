import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_27B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_27B_128K_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "20",
}

QWEN3_6_27B_128K_OTHER_ARGS = [
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
    74000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    6,
    "--max-mamba-cache-size",
    7,
    "--mem-fraction-static",
    0.63,
    "--cuda-graph-bs",
    1,
    2,
    4,
    5,
    6,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
]


class TestNPUQwen3_6_27B_2P_In128k_Out1k_50ms(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.6-27B-w8a8 2p in128k out1k 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    other_args = QWEN3_6_27B_128K_OTHER_ARGS
    envs = QWEN3_6_27B_128K_ENVS
    dataset_name = "random"
    max_concurrency = 4
    num_prompts = 16
    input_len = 128000
    output_len = 1000
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 41.39

    def test_npu_qwen3_6_27b_2p_in128k_out1k_50ms(self):
        """Run NPU performance test for Qwen3.6-27B-w8a8 in128k out1k 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
