import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_30B_A3B_W8A8_MODEL_PATH,
    QWEN3_A3B_EAGLE_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_30B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
}

QWEN3_30B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--max-running-requests",
    16,
    "--disable-radix-cache",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_A3B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    35000,
    "--tp-size",
    2,
    "--mem-fraction-static",
    0.6,
    "--cuda-graph-bs",
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    "--dtype",
    "bfloat16",
]


class TestQwen30B(TestAscendPerformanceTestCaseBase):
    max_attempts = 5
    model = QWEN3_30B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_30B_OTHER_ARGS
    envs = QWEN3_30B_ENVS
    dataset_name = "random"
    max_concurrency = 16
    num_prompts = 16
    input_len = 6144
    output_len = 1500
    random_range_ratio = 1
    tpot = 10.25
    output_token_throughput = 926

    def test_qwen3_30b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
