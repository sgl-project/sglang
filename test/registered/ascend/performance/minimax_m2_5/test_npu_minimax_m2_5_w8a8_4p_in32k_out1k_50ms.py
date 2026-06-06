import os
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

MINIMAX_M2_5_4P_32K_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TASK_QUEUE_ENABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "640",
    "HCCL_BUFFSIZE": "128",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "60184",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBAL_ENABLE_GRAPH": "1",
    "ZBAL_HCCL_OP": "allreduce,_allgather_base,allgather,broadcast,scatter,reduce_scatter,_reduce_scatter_base,alltoall_base",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
}

MINIMAX_M2_5_4P_32K_OTHER_ARGS = [
    "--tp-size",
    8,
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.74,
    "--max-running-requests",
    24,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    32768,
    "--cuda-graph-bs",
    4,
    8,
    12,
    16,
    20,
    24,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--tokenizer-worker-num",
    8,
]


class TestNPUMiniMaxM2_5_W8A8_4P_In32k_Out1k_HighThroughput(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for MiniMax-M2.5-w8a8 4p single node high throughput in32k out1k"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_4P_32K_OTHER_ARGS
    envs = MINIMAX_M2_5_4P_32K_ENVS
    dataset_name = "random"
    max_concurrency = 24
    num_prompts = 96
    input_len = 32768
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 280.44

    def test_npu_minimax_m2_5_w8a8_8p_in32k_out1k_high_throughput(self):
        """Run NPU performance test for MiniMax-M2.5-w8a8 in32k out1k"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
