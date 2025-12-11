import unittest

from sglang.srt.utils import is_npu
from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    NIC_NAME
)

QWEN3_235B_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
QWEN3_235B_OTHER_ARGS = (
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
        "576",
        "--context-length",
        "8192",
        "--dtype",
        "bfloat16",
        "--chunked-prefill-size",
        "102400",
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
        "16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-bs",
        6,
        12,
        18,
        36,
    ]
    if is_npu()
    else []
)

QWEN3_235B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2100",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ENABLE_ASCEND_MOE_NZ": "1",
}

class TestQwen3_235B(TestSingleMixUtils):
    model = QWEN3_235B_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 78
    input_len = 2048
    output_len = 2048
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
