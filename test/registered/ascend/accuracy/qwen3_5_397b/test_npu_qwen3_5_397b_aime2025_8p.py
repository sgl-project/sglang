import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_5_397B_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)

QWEN3_5_397B_W4A8_1P_HIGH_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "0",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "6",
    "DEEPEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "59648",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "SGLANG_ZBAL_BOOTSTRAP_URL": "tcp://127.0.0.1:24669",
    "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBAL_ENABLE_GRAPH": "1",
}

QWEN3_5_397B_W4A8_1P_HIGH_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    17500,
    "--max-total-tokens",
    280000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    432,
    "--mem-fraction-static",
    0.8,
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    50,
    52,
    54,
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
    8,
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


class TestNPUQwen3_5_397B_W4A8_1P_In3k5_Out1k5_High_AIME2025(
    TestAscendAccuracyTestCaseBase
):
    """Test NPU accuracy for Qwen3.5-397B-W4A8 1p in3k5 out1k5 high throughput on AIME2025"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_5_397B_W4A8_MODEL_PATH
    other_args = QWEN3_5_397B_W4A8_1P_HIGH_OTHER_ARGS
    envs = QWEN3_5_397B_W4A8_1P_HIGH_ENVS
    accuracy = 0.933
    datasets = ["aime25"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_npu_qwen3_5_397b_w4a8_1p_in3k5_out1k5_high_aime2025(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
