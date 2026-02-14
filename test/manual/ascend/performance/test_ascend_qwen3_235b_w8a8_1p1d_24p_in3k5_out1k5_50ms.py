import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerfMultiNodePdSepTestCaseBase,
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    QWEN3_235B_W8A8_MODEL_PATH
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME

MODEL_CONFIG = {
    "model_path": QWEN3_235B_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
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
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "DP_ROUND_ROBIN": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
        "HCCL_BUFFSIZE": "512",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
    },
    "router_envs": {
        "SGLANG_DP_ROUND_ROBIN": "1",
    },
    "prefill_args": [
        "--disaggregation-mode", "prefill",
        "--nnodes", "1",
        "--node-rank", "0",
        "--tp-size", 16,
        "--dp-size", 16,
        "--mem-fraction-static", 0.6,
        "--disable-radix-cache",
        # "--ep-dispatch-algorithm",
        # "static",
        # "--init-expert-location",
        # "/hot_map/xxx.pt",
        "--quantization", "modelslim",
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--speculative-draft-model-quantization", "unquant",
        "--max-running-requests", 128,
        "--chunked-prefill-size", 262144,
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
        "--dp-size", 32,
        "--mem-fraction-static", 0.83,
        "--max-running-requests", 768,
        "--quantization", "modelslim",
        "--enable-dp-attention",
        "--moe-a2a-backend", "ascend_fuseep",
        "--cuda-graph-bs", 6, 8, 12, 15, 18, 20, 22, 24,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--speculative-draft-model-quantization", "unquant",
        "--watchdog-timeout", 9000,
        "--context-length", 8192,
        "--prefill-round-robin-balance",
        "--enable-dp-lm-head",
        "--tokenizer-worker-num", 4,
        "--dtype", "bfloat16",
        "--load-balance-method", "round_robin",
    ],
    "router_args": [
        "--mini-lb",
    ],
}


class TestQwen235bW8A8(TestAscendPerfMultiNodePdSepTestCaseBase):
    model_config = MODEL_CONFIG
    backend = "sglang-oai"
    dataset_name = "random"
    max_concurrency = 860
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 35.8
    # T:290@50ms. 800I: 1.8*T
    output_token_throughput = 11570

    def test_throughput(self):
        self.run_throughput(run_cycles=3)


if __name__ == "__main__":
    unittest.main()
