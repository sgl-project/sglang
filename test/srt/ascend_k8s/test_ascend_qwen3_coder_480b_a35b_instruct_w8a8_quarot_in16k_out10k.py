import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
)

QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ENABLE_ASCEND_MOE_NZ": "1",
    "USE_DEEPEP_INT8": "1",
    "STREAMS_PER_DEVICE": "32",
}
QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--tp-size",
        "16",
        "--dp-size",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "32",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "w8a8_int8",
        "--enable-dp-attention",
        "--cuda-graph-bs",
        "8",
        "--watchdog-timeout",
        "9000",
        "--chunked-prefill-size",
        "32768",
        "--max-prefill-tokens",
        "458880",
        "--prefill-round-robin-balance",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--disable-radix-cache",
        "--dtype",
        "bfloat16",
    ]
)

class TestQwen3_Coder_480B_A35b_Instruct_W8a8_Quarot(TestSingleMixUtils):
    model = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_MODEL_PATH
    other_args = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_OTHER_ARGS
    envs = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 32
    input_len = 16000
    output_len = 10000
    random_range_ratio = 1
    ttft = 1206.81
    tpot = 36.45
    output_token_throughput = 252

    def test_qwen3_coder_480b_a35b_instruct_w8a8_quarot(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
