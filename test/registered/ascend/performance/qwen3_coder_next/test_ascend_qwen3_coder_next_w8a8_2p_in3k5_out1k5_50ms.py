import logging
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_CODER_NEXT_W8A8_MODEL_PATH,
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "330",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3000",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "HCCL_BUFFSIZE": "2000",
    "ZBCCL_LOCAL_MEM_SIZE": "60416",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "ZBCCL_BOOTSTRAP_URL": "tcp://127.0.0.1:24669",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ZBCCL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBCCL_ENABLE_GRAPH": "1",
}

OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    4,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.75,
    "--disable-radix-cache",
    "--max-prefill-tokens",
    14080,
    "--context-length",
    26384,
    # "--max-total-tokens",
    # 870000,
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
    312,
    "--cuda-graph-bs",
    2,
    4,
    16,
    32,
    48,
    64,
    80,
    96,
    128,
    140,
    156,
    "--mamba-ssm-dtype",
    "bfloat16",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--speculative-draft-model-path",
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    "--quantization",
    "modelslim",
]


class TestQwen3CoderNext(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_CODER_NEXT_W8A8_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 312
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 4307

    def testQwen3CoderNext(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
