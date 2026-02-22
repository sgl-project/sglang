import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerformanceTestCaseBase,
    QWEN3_480B_W8A8_MODEL_PATH,
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-16-npu-a3", nightly=True)

QWEN3_480B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2100",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_480B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes", "1",
    "--node-rank", "0",
    "--attention-backend", "ascend",
    "--device", "npu",
    "--quantization", "modelslim",
    "--max-running-requests", 80,
    "--context-length", 8192,
    "--dtype", "bfloat16",
    "--chunked-prefill-size", 28672,
    "--max-prefill-tokens", 458880,
    "--disable-radix-cache",
    "--moe-a2a-backend", "deepep",
    "--deepep-mode", "auto",
    "--tp-size", 16,
    "--dp-size", 4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--mem-fraction-static", 0.7,
    "--cuda-graph-bs", 16, 20, 24,
]

class TestQwen480B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_480B_W8A8_MODEL_PATH
    other_args = QWEN3_480B_OTHER_ARGS
    envs = QWEN3_480B_ENVS
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    # T: 143@50ms.   800I: 1.1*T
    output_token_throughput = 1470

    def test_qwen3_480b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
