import unittest

from test_ascend_single_mix_utils import TestSingleNodeTestCaseBase
from test_ascend_single_mix_utils import NIC_NAME

Qwen3_480B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"

Qwen3_480B_ENVS = {
    # "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2100",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

Qwen3_480B_OTHER_ARGS = [
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
    "96",
    "--context-length",
    "8192",
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    "28672",
    "--max-prefill-tokens",
    "458880",
    "--disable-radix-cache",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--tp-size",
    "16",
    "--dp-size",
    "4",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--mem-fraction-static",
    "0.7",
    "--cuda-graph-bs",
    16,
    20,
    24,
]

class TestQwen3_480B(TestSingleNodeTestCaseBase):
    model = Qwen3_480B_MODEL_PATH
    other_args = Qwen3_480B_OTHER_ARGS
    envs = Qwen3_480B_ENVS
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 49.3
    # T: 143@50ms.   800I: 1.1*T
    output_token_throughput = 1490

    def test_qwen3_480b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
