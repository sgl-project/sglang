import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_NEXT_80B_A3B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "400",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "HCCL_BUFFSIZE": "2000",
    "ZBCCL_LOCAL_MEM_SIZE": "60416",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "ZBCCL_BOOTSTRAP_URL": "tcp://127.0.0.1:24669",
    "ZBCCL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBCCL_ENABLE_GRAPH": "1",
}

QWEN3_NEXT_80B_A3B_OTHER_ARGS = [
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
    0.8,
    "--disable-radix-cache",
    "--max-prefill-tokens",
    28672,
    "--context-length",
    26384,
    "--max-total-tokens",
    870000,
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
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    16,
    "--cuda-graph-bs",
    2,
    4,
    8,
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-draft-model-path",
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
]


class TestQwen3Next80BA3B(TestAscendPerformanceTestCaseBase):
    max_attempts = 5
    model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_NEXT_80B_A3B_OTHER_ARGS
    envs = QWEN3_NEXT_80B_A3B_ENVS
    dataset_name = "random"
    max_concurrency = 16
    num_prompts = 16
    input_len = 1000
    output_len = 300
    random_range_ratio = 1
    tpot = 14.21
    output_token_throughput = 760

    def test_qwen3_next_80b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
