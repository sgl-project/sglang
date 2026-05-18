import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_480B_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdMixTestCaseBase,
)

MODEL_CONFIG = {
    "model_path": QWEN3_480B_W8A8_MODEL_PATH,
    "node_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "72",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "HCCL_BUFFSIZE": "1800",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_OP_EXPANSION_MODE": "AIV",
    },
    "other_args": [
        "--trust-remote-code",
        "--nnodes",
        "2",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--max-running-requests",
        288,
        "--context-length",
        8192,
        "--dtype",
        "bfloat16",
        "--chunked-prefill-size",
        114688,
        "--max-prefill-tokens",
        458880,
        "--disable-radix-cache",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--tp-size",
        32,
        "--dp-size",
        4,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static",
        0.7,
        "--cuda-graph-bs",
        56,
        64,
        72,
    ],
}


class TestQwen480B(TestAscendPerfMultiNodePdMixTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 288
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 48.9
    output_token_throughput = 4880

    def test_qwen3_480b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
