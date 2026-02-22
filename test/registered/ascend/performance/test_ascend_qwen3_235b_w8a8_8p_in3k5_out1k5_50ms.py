import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerformanceTestCaseBase,
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    QWEN3_235B_W8A8_MODEL_PATH
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-16-npu-a3", nightly=True)

QWEN3_235B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "2",
}

QWEN3_235B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes", "1",
        "--node-rank", "0",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--max-running-requests", 272,
        "--context-length", 8192,
        "--dtype", "bfloat16",
        "--chunked-prefill-size", 32768,
        "--max-prefill-tokens", 32768,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--disable-radix-cache",
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "auto",
        "--speculative-draft-model-quantization", "unquant",
        "--tp-size", 16,
        "--dp-size", 16,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static", 0.8,
        "--cuda-graph-bs", 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17,
    ]
)

class TestQwen235B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_235B_W8A8_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    max_concurrency = 272
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 48.5
    # T: 290@50ms.   800I: 1.1*T
    output_token_throughput = 4087

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
