import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_8B_MODEL_PATH,
    QWEN3_8B_OTHER_ARGS,
    QWEN3_8B_ENVS,
)

QWEN3_8B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",     
    "HCCL_OP_EXPANSION_MODE": "AIV",
    
}

QWEN3_8B_OTHER_ARGS = (
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
        "w8a8_int8",
        "--max-running-requests",
        "16",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "43008",
        "--max-prefill-tokens",
        "525000",
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-bs",
        "16",
        "--dtype",
        "bfloat16",    
    ]
)


class TestQwen3_8B(TestSingleMixUtils):
    model = QWEN3_8B_MODEL_PATH
    other_args = QWEN3_8B_OTHER_ARGS
    envs = QWEN3_8B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 8
    input_len = 1000
    output_len = 300
    random_range_ratio = 0.5
    ttft = 291.95
    tpot = 18.83
    output_token_throughput = 404.29

    def test_qwen3_8b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
