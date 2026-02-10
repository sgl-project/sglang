import unittest

from sglang.test.ascend.performance.test_ascend_performance_utils import (
    TestPerformanceTestCaseBase,
    NIC_NAME,
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    QWEN3_235B_MODEL_PATH
)
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
}

QWEN3_235B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes", "1",
        "--node-rank", "0",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--max-running-requests", 1,
        "--dtype", "bfloat16",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 16384,
        "--speculative-draft-model-quantization", "unquant",
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 4,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 5,
        "--disable-radix-cache",
        "--enable-dp-lm-head",
        "--tp-size", 16,
        "--mem-fraction-static", 0.78,
        "--cuda-graph-bs", 1,
    ]
)

class TestQwen235B(TestPerformanceTestCaseBase):
    model = QWEN3_235B_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 11000
    output_len = 1000
    random_range_ratio = 1
    tpot = 9.24
    # T: None.   800I: None        Dev: 93.52/8@9.7ms
    output_token_throughput = 99

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
