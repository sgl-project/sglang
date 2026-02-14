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
    "HCCL_BUFFSIZE": "1200",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "144",
}

QWEN3_235B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes", "1",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--max-running-requests", 576,
        "--context-length", 8192,
        "--dtype", "bfloat16",
        "--chunked-prefill-size", 32768,
        "--max-prefill-tokens", 458880,
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
        "--mem-fraction-static", 0.84,
        "--cuda-graph-bs", 8, 16, 20, 24, 32, 36,
    ]
)

class TestQwen235B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_235B_W8A8_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    max_concurrency = 576
    num_prompts = 576
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    tpot = 50
    # T: 320@100ms      800I:1.8*T         Dev-800I: 6320/8@54.78
    output_token_throughput = 6856

    def test_qwen3_235b(self):
        self.run_throughput(run_cycles=4)


if __name__ == "__main__":
    unittest.main()
