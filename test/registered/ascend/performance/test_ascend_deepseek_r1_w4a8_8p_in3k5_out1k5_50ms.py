import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerformanceTestCaseBase,
    DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-16-npu-a3", nightly=True)

MODEL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "36",
    "HCCL_BUFFSIZE": "1600",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "ENABLE_MOE_NZ": "1",
}

MODEL_OTHER_ARGS = (
    [
        "--tp-size", 16,
        "--trust-remote-code",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--watchdog-timeout", "9000",
        "--cuda-graph-bs", 8, 16, 24, 28, 32, 36,
        "--mem-fraction-static", 0.71,
        "--max-running-requests", 144,
        "--context-length", 8188,
        "--disable-radix-cache",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 9000,
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "auto",
        "--enable-dp-attention",
        "--dp-size", 4,
        "--enable-dp-lm-head",
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--dtype", "bfloat16",
    ]
)


class TestAscendDeepSeekR1W4A8(TestAscendPerformanceTestCaseBase):
    model = DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH
    other_args = MODEL_OTHER_ARGS
    envs = MODEL_ENVS
    dataset_name = "random"
    max_concurrency = 144
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 44.9
    # T: 146@50ms. 800I A3：1.1*T
    output_token_throughput = 2761

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
