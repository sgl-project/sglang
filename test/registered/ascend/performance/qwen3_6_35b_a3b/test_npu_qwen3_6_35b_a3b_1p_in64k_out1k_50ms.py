import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_35B_A3B_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-2-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_35B_A3B_64K_1K_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "1",
}

QWEN3_6_35B_A3B_64K_1K_OTHER_ARGS = [
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
    "--max-total-tokens",
    600000,
    "--max-prefill-tokens",
    65536,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests",
    10,
    "--max-mamba-cache-size",
    20,
    "--mem-fraction-static",
    0.65,
    "--cuda-graph-bs",
    2,
    4,
    8,
    12,
    14,
    16,
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


class TestNPUQwen3_6_35BA3B_1P_In64k_Out1k_50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.6-35B-A3B 1p in64k out1k 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_6_35B_A3B_MODEL_PATH
    other_args = QWEN3_6_35B_A3B_64K_1K_OTHER_ARGS
    envs = QWEN3_6_35B_A3B_64K_1K_ENVS
    dataset_name = "random"
    max_concurrency = 10
    num_prompts = 40
    input_len = 64000
    output_len = 1000
    random_range_ratio = 1
    seed = 1
    tpot = 50
    output_token_throughput = 141.72

    def test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_50ms(self):
        """Run NPU performance test for Qwen3.6-35B-A3B in64k out1k 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
