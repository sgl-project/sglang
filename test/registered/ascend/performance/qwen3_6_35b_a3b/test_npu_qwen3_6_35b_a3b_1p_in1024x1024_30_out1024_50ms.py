import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_35B_A3B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_35B_A3B_1024_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_VIT_ENABLE_CUDA_GRAPH": "1",
}

QWEN3_6_35B_A3B_1024_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    9999999,
    "--max-total-tokens",
    365000,
    "--prefill-max-requests",
    30,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    120,
    "--max-mamba-cache-size",
    120,
    "--mem-fraction-static",
    0.85,
    "--cuda-graph-bs",
    4,
    16,
    32,
    48,
    64,
    110,
    165,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
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


class TestNPUQwen3_6_35BA3B_1P_In1024x1024_30_Out1024_50ms(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for Qwen3.6-35B-A3B 1p in1024x1024 30 out1024 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    model = QWEN3_6_35B_A3B_MODEL_PATH
    other_args = QWEN3_6_35B_A3B_1024_OTHER_ARGS
    envs = QWEN3_6_35B_A3B_1024_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    max_concurrency = 120
    num_prompts = 120
    input_len = 30
    output_len = 1024
    random_range_ratio = 1
    image_resolution = "1024x1024"
    image_count = 1
    seed = 1
    tpot = 50
    request_rate = float("inf")
    output_token_throughput = 1620

    def test_npu_qwen3_6_35b_a3b_1p_in1024x1024_30_out1024_50ms(self):
        """Run NPU performance test for Qwen3.6-35B-A3B in1024x1024 30 out1024 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
