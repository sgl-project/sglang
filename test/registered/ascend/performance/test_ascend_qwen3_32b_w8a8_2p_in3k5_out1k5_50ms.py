import unittest

from test_ascend_single_mix_utils import (
    TestSingleNodeTestCaseBase,
    NIC_NAME
)

QWEN3_32B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
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
        "--quantization",
        "modelslim",
        "--max-running-requests",
        "120",
        "--disable-radix-cache",
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_32B_EAGLE_MODEL_PATH,
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "49152",
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-bs",
        16,
        32,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        "--dtype",
        "bfloat16",
    ]
)


class TestQwen3_32B(TestSingleNodeTestCaseBase):
    model = QWEN3_32B_W8A8_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    max_concurrency = 78
    num_prompts = 312
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    # T: 387. 800I A3: 1.8*T=696.6
    output_token_throughput = 1390

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
