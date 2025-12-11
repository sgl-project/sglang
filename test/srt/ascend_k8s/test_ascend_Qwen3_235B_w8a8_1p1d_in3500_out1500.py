import unittest

from test_ascend_disaggregation_utils import (
    TestAscendDisaggregationUtils,
    NIC_NAME
)

MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "HCCL_BUFFSIZE": "3000",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
        "ENABLE_ASCEND_MOE_NZ": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    },
    "decode_envs": {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "DP_ROUND_ROBIN": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "30",
        "HCCL_BUFFSIZE": "512",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
    },
    "prefill_args": [
        "--disaggregation-mode",
        "prefill",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--tp-size",
        16,
        "--dp-size",
        16,
        "--mem-fraction-static",
        0.6,
        "--disable-radix-cache",
        "--quantization",
        "modelslim",
        "--max-running-requests",
        128,
        "--chunked-prefill-size",
        114688,
        "--max-prefill-tokens",
        458880,
        "--disable-overlap-schedule",
        "--enable-dp-attention",
        "--tokenizer-worker-num",
        4,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--dtype",
        "bfloat16",
    ],
    "decode_args": [
        "--disaggregation-mode",
        "decode",
        "--nnodes",
        "2",
        "--tp-size",
        32,
        "--dp-size",
        32,
        "--mem-fraction-static",
        0.83,
        "--max-running-requests",
        960,
        "--quantization",
        "modelslim",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--cuda-graph-bs",
        6,
        8,
        12,
        15,
        18,
        20,
        22,
        24,
        27,
        30,
        "--watchdog-timeout",
        9000,
        "--context-length",
        8192,
        "--prefill-round-robin-balance",
        "--enable-dp-lm-head",
        "--tokenizer-worker-num",
        4,
        "--dtype",
        "bfloat16",
    ],
}


class TestQwen3_235B_w8a8_1p2d_in3500_out1500(TestAscendDisaggregationUtils):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 16
    max_concurrency = 20
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 9073.9
    tpot = 63.7
    output_token_throughput = 470.07

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
