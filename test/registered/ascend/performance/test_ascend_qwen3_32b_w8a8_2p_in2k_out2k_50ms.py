import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerformanceTestCaseBase,
    QWEN3_32B_EAGLE_MODEL_PATH,
    QWEN3_32B_W8A8_MODEL_PATH
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-4-npu-a3", nightly=True)

QWEN3_32B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
}

QWEN3_32B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes", "1",
        "--node-rank", "0",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--max-running-requests", 120,
        "--disable-radix-cache",
        "--speculative-draft-model-quantization", "unquant",
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_32B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 49152,
        "--tp-size", 4,
        "--mem-fraction-static", 0.7,
        "--cuda-graph-bs", 54, 60, 66, 72, 78, 84, 90, 108, 114, 120,
        "--dtype", "bfloat16",
    ]
)


class TestQwen32B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_32B_W8A8_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    max_concurrency = 120
    num_prompts = 480
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    tpot = 33.5
    # T: 472/@64ms. 800I A3：1972.3
    output_token_throughput = 1960

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
