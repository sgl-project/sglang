import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    NIC_NAME
)

QWQ_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/QWQ-32B-W8A8"
QWQ_32B_OTHER_ARGS = [
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
    "16",
    "--context-length",
    "8192",
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    "32768",
    "--max-prefill-tokens",
    "458880",
    "--disable-radix-cache",
    "--tp-size",
    "4",
    "--enable-dp-lm-head",
    "--mem-fraction-static",
    "0.68",
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    10,
    12,
    16,
]

QWQ_32B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2048",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

class TestQWQ_32B(TestSingleMixUtils):
    model = QWQ_32B_MODEL_PATH
    other_args = QWQ_32B_OTHER_ARGS
    envs = QWQ_32B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 8
    input_len = 6144
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 22.63
    output_token_throughput = 300

    def test_qwq_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
