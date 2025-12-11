import unittest

from sglang.srt.utils import is_npu
from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    NIC_NAME
)

QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
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
        "78",
        "--context-length",
        "8192",
        "--enable-hierarchical-cache",
        "--hicache-write-policy",
        "write_through",
        "--hicache-ratio",
        "3",
        "--chunked-prefill-size",
        "43008",
        "--max-prefill-tokens",
        "52500",
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.68",
        "--cuda-graph-bs",
        "78",
        "--dtype",
        "bfloat16"
    ]
    if is_npu()
    else []
)

QWEN3_32B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}


class TestQwen3_32B(TestSingleMixUtils):
    model = QWEN3_32B_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 32
    input_len = 6144
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 38.37
    output_token_throughput = 800

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
