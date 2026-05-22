import os
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

MINIMAX_M2_5_HIGH_THROUGHPUT_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "800",
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
    480,
    "--disable-radix-cache",
    "--prefill-delayer-max-delay-passes",
    500,
    "--enable-prefill-delayer",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    8192,
    "--cuda-graph-bs",
    4,
    24,
    32,
    48,
    64,
    80,
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
]


class TestNPUMiniMaxM2_5_W8A8_8P_GPQA(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for MiniMax-M2.5-w8a8 8p single node high throughput on GPQA"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_HIGH_THROUGHPUT_OTHER_ARGS
    envs = MINIMAX_M2_5_HIGH_THROUGHPUT_ENVS
    accuracy = 85.2
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    max_concurrency = 64
    generation_kwargs = "dict(temperature=1.0)"
    output_len = 65536

    def test_npu_minimax_m2_5_w8a8_8p_gpqa(self):
        """Run NPU accuracy test for MiniMax-M2.5-w8a8 8p single node high throughput on GPQA"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
