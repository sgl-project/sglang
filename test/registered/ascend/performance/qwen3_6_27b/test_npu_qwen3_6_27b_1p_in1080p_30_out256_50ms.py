import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_27B_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-2-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_27B_1080P_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_VIT_ENABLE_CUDA_GRAPH": "1",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
}

QWEN3_6_27B_1080P_OTHER_ARGS = [
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
    82688,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    38,
    "--max-mamba-cache-size",
    38,
    "--mem-fraction-static",
    0.70,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    10,
    12,
    16,
    20,
    24,
    28,
    30,
    32,
    35,
    38,
    "--enable-prefill-delayer",
    "--prefill-delayer-queue-min-ratio",
    0.45,
    "--perfill-delayer-max-delay-ms",
    5500,
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


class TestNPUQwen3_6_27B_1P_In1080p_30_Out256_50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.6-27B 1p in1080p 30 out256 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    model = QWEN3_6_27B_MODEL_PATH
    other_args = QWEN3_6_27B_1080P_OTHER_ARGS
    envs = QWEN3_6_27B_1080P_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    warmup_requests = 38
    max_concurrency = 42
    num_prompts = 152
    input_len = 30
    output_len = 256
    random_range_ratio = 1
    image_resolution = "1920x1080"
    image_count = 1
    seed = 1
    tpot = 50
    output_token_throughput = 226

    def test_npu_qwen3_6_27b_1p_in1080p_30_out256_50ms(self):
        """Run NPU performance test for Qwen3.6-27B in1080p 30 out256 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
