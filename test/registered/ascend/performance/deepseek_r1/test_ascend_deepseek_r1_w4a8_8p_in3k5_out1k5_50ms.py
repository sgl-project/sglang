import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

MODEL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "56",
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_USE_FIA_NZ": "1",
}

MODEL_OTHER_ARGS = [
    "--tp-size",
    16,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--watchdog-timeout",
    "9000",
    "--cuda-graph-bs",
    4,
    8,
    12,
    14,
    "--mem-fraction-static",
    0.77,
    "--max-running-requests",
    224,
    "--context-length",
    8188,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    3000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--dtype",
    "bfloat16",
]


class TestAscendDeepSeekR1W4A8(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    # aisbench_dataset_path = "/data/c30044170/dataset/GSM8K-in3584-bs7168.jsonl"
    model = DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH
    other_args = MODEL_OTHER_ARGS
    envs = MODEL_ENVS
    dataset_name = "random"
    max_concurrency = 224
    num_prompts = 896
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50.36
    output_token_throughput = 3547

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
