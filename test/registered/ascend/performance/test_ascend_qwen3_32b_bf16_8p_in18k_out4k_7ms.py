import unittest

from test_ascend_single_mix_utils import (
    TestSingleNodeTestCaseBase,
    NIC_NAME
)

QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
QWEN3_32B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B-Eagle3"

QWEN3_32B_ENVS = {
    # "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DISABLE_EAGLE3_QUANT": "1",
}

QWEN3_32B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--max-running-requests",
        "32",
        "--disable-radix-cache",
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_32B_EAGLE_MODEL_PATH,
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "3",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "65536",
        "--tp-size",
        "16",
        "--mem-fraction-static",
        "0.72",
        "--cuda-graph-bs",
        1,
        4,
        6,
        12,
        18,
        24,
        30,
        32,
        "--dtype",
        "bfloat16",
    ]
)


class TestQwen3_32B(TestSingleNodeTestCaseBase):
    model = QWEN3_32B_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 18000
    output_len = 4000
    random_range_ratio = 1
    tpot = 14.6
    # 800I A3：79.64
    output_token_throughput = 67.2

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
