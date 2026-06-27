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
    disabled="performance case",
)

QWEN3_6_27B_1024_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_VIT_ENABLE_CUDA_GRAPH": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NPU_PROFILING": "1",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "150",
    "ASCEND_USE_FIA": "1",
}

QWEN3_6_27B_1024_OTHER_ARGS = [
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
    52000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    50,
    "--max-mamba-cache-size",
    60,
    "--mem-fraction-static",
    0.76,
    "--cuda-graph-bs",
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    42,
    45,
    50,
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
    "--mm-enable-dp-encoder",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]


class TestNPUQwen3_6_27B_1P_In1024x1024_30_Out1024_50ms(
    TestNpuPerformanceTestCaseBase
):
    """Test NPU performance for Qwen3.6-27B 1p in1024x1024 30 out1024 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    model = QWEN3_6_27B_MODEL_PATH
    other_args = QWEN3_6_27B_1024_OTHER_ARGS
    envs = QWEN3_6_27B_1024_ENVS
    dataset_name = "random"
    max_concurrency = 48
    num_prompts = 48
    input_len = 30
    output_len = 1024
    random_range_ratio = 1
    image_resolution = "1024x1024"
    image_count = 1
    tpot = 50
    output_token_throughput = 800.8

    def test_npu_qwen3_6_27b_1p_in1024x1024_30_out1024_50ms(self):
        """Run NPU performance test for Qwen3.6-27B in1024x1024 30 out1024 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
