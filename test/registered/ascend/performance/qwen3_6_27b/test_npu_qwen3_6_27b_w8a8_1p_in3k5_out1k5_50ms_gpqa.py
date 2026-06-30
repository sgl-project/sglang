import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_27B_W8A8_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-2-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_27B_3K5_1K5_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "130",
    "ASCEND_USE_FIA": "1",
}

QWEN3_6_27B_3K5_1K5_OTHER_ARGS = [
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
    60000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    64,
    "--max-mamba-cache-size",
    74,
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs",
    2,
    8,
    16,
    32,
    48,
    64,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
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
]


class TestNPUQwen3_6_27B_1P_In3k5_Out1k5_50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.6-27B-w8a8 1p in3k5 out1k5 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    other_args = QWEN3_6_27B_3K5_1K5_OTHER_ARGS
    envs = QWEN3_6_27B_3K5_1K5_ENVS
    dataset_name = "random"
    max_concurrency = 54
    num_prompts = 216
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 786.69

    def test_npu_qwen3_6_27b_1p_in3k5_out1k5_50ms(self):
        """Run NPU performance test for Qwen3.6-27B-w8a8 in3k5 out1k5 50ms"""
        self.run_throughput()


class TestNPUQwen3_6_27B_1P_In3k5_Out1k5_gpqa(TestNpuAccuracyTestCaseBase):
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    envs = QWEN3_6_27B_3K5_1K5_ENVS
    other_args = QWEN3_6_27B_3K5_1K5_OTHER_ARGS
    accuracy = 0.855
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    eval_batch_size = 8
    generation_config = {"max_tokens": 81920, "temperature": 1.0}

    def test_accuracy(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
