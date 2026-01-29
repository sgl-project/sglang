import unittest

from sglang.test.ascend.performance.test_ascend_performance_utils import (
    TestPerformanceTestCaseBase,
    NIC_NAME,
    QWEN3_30B_A3B_W8A8_MODEL_PATH,
    QWEN3_A3B_EAGLE_MODEL_PATH
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-2-npu-a3", nightly=True)

QWEN3_30B_A3B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
}

QWEN3_30B_A3B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes", "1",
        "--node-rank", "0",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--max-running-requests", 192,
        "--disable-radix-cache",
        "--speculative-draft-model-quantization", "unquant",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 32768,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_A3B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--tp-size", 2,
        "--mem-fraction-static", 0.86,
        "--cuda-graph-bs", 42, 88, 96, 132, 144, 156, 172, 178, 192,
        "--dtype", "bfloat16",
    ]
)

class TestQwen30B(TestPerformanceTestCaseBase):
    model = QWEN3_30B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_30B_A3B_OTHER_ARGS
    envs = QWEN3_30B_A3B_ENVS
    dataset_name = "random"
    max_concurrency = 156
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 45.6
    # T: 1493@51ms       800I: 1.8*T        Dev-800I: 3166@44.35ms
    output_token_throughput = 2960

    def test_qwen3_30b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
