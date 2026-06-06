import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_NEXT_80B_A3B_2P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "330",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3000",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "ENABLE_PROFILING": "0",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "HCCL_BUFFSIZE": "64",
    "ZBAL_HCCL_OP": "allreduce,_allgather_base,allgather,broadcast,scatter,reduce_scatter,_reduce_scatter_base,alltoall_base",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "59648",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "SGLANG_ZBAL_BOOTSTRAP_URL": "tcp://127.0.0.1:24669",
    "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBAL_ENABLE_GRAPH": "1",
}

QWEN3_NEXT_80B_A3B_2P_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--page-size",
    128,
    "--tp-size",
    4,
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.75,
    "--disable-radix-cache",
    "--max-prefill-tokens",
    14080,
    "--context-length",
    26384,
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    300,
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
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-draft-model-path",
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    150,
]


class TestNPUQwen3Next80BA3B2PIn3k5Out1k5_50ms(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_NEXT_80B_A3B_2P_OTHER_ARGS
    envs = QWEN3_NEXT_80B_A3B_2P_ENVS
    dataset_name = "random"
    max_concurrency = 220
    num_prompts = 220
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 4293

    def test_npu_qwen3_next_80b_a3b_2p_in3k5_out1k5_50ms(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
