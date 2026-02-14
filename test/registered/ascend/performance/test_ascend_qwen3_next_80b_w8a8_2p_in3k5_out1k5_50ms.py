import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerformanceTestCaseBase,
    QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-4-npu-a3", nightly=True)

QWEN3_NEXT_80B_A3B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "20",
    "HCCL_BUFFSIZE": "2000",
}

QWEN3_NEXT_80B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend", "ascend",
    "--device", "npu",
    "--tp-size", 4,
    "--mem-fraction-static", 0.685,
    "--max-running-requests", 80,
    "--watchdog-timeout", 9000,
    "--disable-radix-cache",
    "--cuda-graph-bs", 80,
    "--max-prefill-tokens", 28672,
    "--max-total-tokens", 450560,
    "--moe-a2a-backend", "deepep",
    "--deepep-mode", "auto",
    "--quantization", "modelslim",
    "--chunked-prefill-size", -1,
]

class TestQwen3Next80BA3B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_NEXT_80B_A3B_OTHER_ARGS
    envs = QWEN3_NEXT_80B_A3B_ENVS
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = 320
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    # T: 1976@50ms       800I A3: None      Dev-800I: 1405.17/2 @49.91ms
    output_token_throughput = 1410

    def test_qwen3_next_80b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
