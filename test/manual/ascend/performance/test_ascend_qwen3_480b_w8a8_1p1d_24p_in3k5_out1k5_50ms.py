import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerfMultiNodePdSepTestCaseBase,
    QWEN3_480B_W8A8_MODEL_PATH
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME

MODEL_CONFIG = {
    "model_path": QWEN3_480B_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
        "HCCL_BUFFSIZE": "4300",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "72",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "HCCL_BUFFSIZE": "512",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
    },
    "prefill_args": [
        "--disaggregation-mode", "prefill",
        "--nnodes", "1",
        "--node-rank", "0",
        "--tp-size", 16,
        "--dp-size", 2,
        "--mem-fraction-static", 0.6,
        "--disable-radix-cache",
        "--quantization", "modelslim",
        "--max-running-requests", 128,
        "--chunked-prefill-size", 65536,
        "--max-prefill-tokens", 262144,
        "--enable-dp-attention",
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "normal",
        "--dtype", "bfloat16",
    ],
    "decode_args": [
        "--disaggregation-mode", "decode",
        "--nnodes", "2",
        "--tp-size", 32,
        "--dp-size", 4,
        "--mem-fraction-static", 0.73,
        "--max-running-requests", 384,
        "--quantization", "modelslim",
        "--enable-dp-attention",
        "--moe-a2a-backend", "ascend_fuseep",
        "--cuda-graph-bs", 16, 32, 48, 56, 64, 72, 80, 88, 96,
        "--watchdog-timeout", 9000,
        "--context-length", 8192,
        "--prefill-round-robin-balance",
        "--enable-dp-lm-head",
        "--tokenizer-worker-num", 4,
        "--dtype", "bfloat16",
        "--load-balance-method", "round_robin",
    ],
    "router_args": [
    ],
}


class TestQwen480bW8a8(TestAscendPerfMultiNodePdSepTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 410
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 46.7
    # T:143@50ms. 800I: None     Dev-800I: 6390/24@48.27ms
    output_token_throughput = 6770

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
