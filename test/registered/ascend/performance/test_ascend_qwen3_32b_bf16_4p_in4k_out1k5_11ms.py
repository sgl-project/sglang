import unittest

from sglang.test.ascend.performance.test_ascend_performance_utils import (
    TestPerformanceTestCaseBase,
    NIC_NAME, QWEN3_32B_EAGLE_MODEL_PATH,
    QWEN3_32B_MODEL_PATH
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-8-npu-a3", nightly=True)

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
        "--max-running-requests", 1,
        "--disable-radix-cache",
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_32B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 4,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 5,
        "--chunked-prefill-size", 24576,
        "--max-prefill-tokens", 65536,
        "--tp-size", 8,
        "--mem-fraction-static", 0.72,
        "--cuda-graph-bs", 1,
        "--dtype", "bfloat16",
    ]
)


class TestQwen32B(TestPerformanceTestCaseBase):
    model = QWEN3_32B_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 4
    input_len = 4096
    output_len = 1500
    random_range_ratio = 1
    tpot = 8.79
    # 800I A3: 103.40
    output_token_throughput = 120

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
