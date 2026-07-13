import os
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-16-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

MINIMAX_M2_5_HIGH_THROUGHPUT_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "1024",
    "ASCEND_USE_FIA": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "204800",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
}

MINIMAX_M2_5_HIGH_THROUGHPUT_OTHER_ARGS = [
    "--tp-size",
    16,
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--mem-fraction-static",
    0.75,
    "--max-running-requests",
    320,
    "--disable-radix-cache",
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
    "--prefill-delayer-max-delay-passes",
    500,
    "--enable-prefill-delayer",
    "--chunked-prefill-size",
    196608,
    "--max-prefill-token",
    8192,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    12,
    16,
    20,
    "--moe-a2a-backend",
    "ascend_fuseep",
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
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
]


class TestNPUMiniMaxM2_5_W8A8_8P_In3k5_Out1k5_HighThroughput(
    TestNpuPerformanceTestCaseBase
):
    """Test NPU performance for MiniMax-M2.5-w8a8 8p single node high throughput in3k5 out1k5"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_HIGH_THROUGHPUT_OTHER_ARGS
    envs = MINIMAX_M2_5_HIGH_THROUGHPUT_ENVS
    dataset_name = "random"
    max_concurrency = 320
    num_prompts = 1280
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    seed = 1
    tpot = 50
    output_token_throughput = 5717.58

    def test_npu_minimax_m2_5_w8a8_8p_in3k5_out1k5_high_throughput(self):
        """Run NPU performance test for MiniMax-M2.5-w8a8 high throughput"""
        self.run_throughput()


class TestNPUMiniMaxM2_5_W8A8_8P_In3k5_Out1k5_GPQA(TestNpuAccuracyTestCaseBase):
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    envs = MINIMAX_M2_5_HIGH_THROUGHPUT_ENVS
    other_args = MINIMAX_M2_5_HIGH_THROUGHPUT_OTHER_ARGS
    accuracy = 0.852
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_accuracy(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
