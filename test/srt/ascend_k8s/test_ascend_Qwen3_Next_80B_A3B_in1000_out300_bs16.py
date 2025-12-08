import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    NIC_NAME
)

Qwen3_Next_80B_A3B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
Qwen3_Next_80B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    "8",
    "--context-length",
    "8000",
    "--chunked-prefill-size",
    "6400",
    "--max-prefill-tokens",
    "8000",
    "--max-total-tokens",
    "16000",
    "--disable-radix-cache",
    "--tp-size",
    "4",
    "--dp-size",
    "1",
    "--mem-fraction-static",
    "0.78",
    "--cuda-graph-bs",
    1,
    2,
    6,
    8,
    16,
    32,
]

Qwen3_Next_80B_A3B_ENVS = {
    "ASCEND_RT_VISIBLE_DEVICES": "12,13,14,15",
    "SGLANG_SET_CPU_AFFINITY": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "HCCL_BUFFSIZE": "2000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_ALGO": "level0:NA;level1:ring"
}

class TestQwen3_Next_80B_A3B(TestSingleMixUtils):
    model = Qwen3_Next_80B_A3B_MODEL_PATH
    other_args = Qwen3_Next_80B_A3B_OTHER_ARGS
    envs = Qwen3_Next_80B_A3B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 16
    input_len = 1000
    output_len = 300
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_next_80b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
