import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_397B_W4A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_5_397B_64K_PREFIX_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "0",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "20",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "58672",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "SGLANG_ZBAL_BOOTSTRAP_URL": "tcp://127.0.0.1:24669",
    "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBAL_ENABLE_GRAPH": "1",
}

QWEN3_5_397B_64K_PREFIX_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    65536,
    "--max-mamba-cache-size",
    640,
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--trust-remote-code",
    "--max-running-requests",
    128,
    "--mem-fraction-static",
    0.6,
    "--max-total-tokens",
    1310720,
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    10,
    12,
    16,
    20,
    24,
    32,
    40,
    48,
    56,
    64,
    "--quantization",
    "modelslim",
    "--enable-multimodal",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
]


class TestNPUQwen3_5_397B_64K_Prefix90(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.5-397B-w4a8 8p in64k out1k prefix90"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_397B_W4A8_MODEL_PATH
    other_args = QWEN3_5_397B_64K_PREFIX_OTHER_ARGS
    envs = QWEN3_5_397B_64K_PREFIX_ENVS
    dataset_name = "random"
    max_concurrency = 128
    num_prompts = 128
    aisbench_repeat_rate = 0.9
    input_len = 65536
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    aisbench_request_rate = 6
    output_token_throughput = 1012.3

    def test_npu_qwen3_5_397b_8p_in64k_out1k_prefix90_50ms(self):
        """Run NPU performance test for Qwen3.5-397B-w4a8 8p in64k out1k prefix90"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
