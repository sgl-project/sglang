import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_FP4_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-acc-4-npu-a5",
    nightly=True,
)

DEEPSEEK_V4_FLASH_FP4_4P_A5_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "2048",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_INTRA_DMA_ENABLE": "1",
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
}

DEEPSEEK_V4_FLASH_FP4_4P_A5_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    4,
    "--trust-remote-code",
    "--attention-backend",
    "dsv4",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.72,
    "--max-running-requests",
    64,
    "--chunked-prefill-size",
    65536,
    "--max-prefill-tokens",
    65536,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    10,
    16,
    "--kv-cache-dtype",
    "auto",
    "--enable-dp-lm-head",
    "--disable-radix-cache",
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--reasoning-parser",
    "deepseek-v4",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
]

DEEPSEEK_V4_FLASH_FP4_GENERATION_CONFIG_HIGH = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}


class TestNPUDeepSeekV4FlashFP44PGPQA5(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for DeepSeek-V4-Flash FP4 4p A5 GPQA."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = DEEPSEEK_V4_FLASH_FP4_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_FP4_4P_A5_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_FP4_4P_A5_ENVS
    accuracy = 0.874
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = DEEPSEEK_V4_FLASH_FP4_GENERATION_CONFIG_HIGH
    eval_batch_size = 128
    stream = True
    timeout = 6000
    seed = 1

    def test_npu_deepseek_v4_flash_fp4_4p_gpqa_a5(self):
        """Run NPU accuracy test for DeepSeek-V4-Flash FP4 4p A5 GPQA."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
