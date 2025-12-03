import unittest

from test_ascend_disaggregation_utils import (
    TestAscendDisaggregationUtils,
)

MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFIMITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        "SGLANG_USE_AG_AFTER_QLORA": "1",
        "HCCL_BUFFSIZE": "1536",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": "enp23s0f3",
        "GLOO_SOCKET_IFNAME": "enp23s0f3",
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFIMITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        # "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        # "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_BUFFSIZE": "600",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
        "TASK_QUEUE_ENABLE": "0",
        "HCCL_SOCKET_IFNAME": "enp23s0f3",
        "GLOO_SOCKET_IFNAME": "enp23s0f3",    
    },
    "prefill_args": [
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.81,
        "--quantization",
        "w8a8_int8",
        "--max-running-requests",
        8,
        "--context-length",
        8192,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        28680,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        # "--speculative-algorithm",
        # "NEXTN",
        # "--speculative-num-steps",
        # 1,
        # "--speculative-eagle-topk",
        # 1,
        # "--speculative-num-draft-tokens",
        # 2,
        "--dp-size",
        2,
        "--enable-dp-attention",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
    ],
    "decode_args": [
        "--nnodes",
        "2",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        32,
        "--dp-size",
        32,
        "--mem-fraction-static",
        0.8,
        "--max-running-requests",
        832,
        "--quantization",
        "w8a8_int8",
        "--moe-a2a-backend",
        "deepep",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--cuda-graph-bs",
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        "--watchdog-timeout",
        9000,
        "--context-length",
        8192,
        # "--speculative-algorithm",
        # "NEXTN",
        # "--speculative-num-steps",
        # 2,
        # "--speculative-eagle-topk",
        # 1,
        # "--speculative-num-draft-tokens",
        # 3,
        "--tokenizer-worker-num",
        4,
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--prefill-round-robin-balance",  
    ],
}


class Test_DeepSeek_R1_W8A8_2P1D_In2048_Out2048(TestAscendDisaggregationUtils):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 16
    max_concurrency = 8
    num_prompts = int(max_concurrency) * 4
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    ttft = 3167.46
    tpot = 46.18
    output_token_throughput = 12663.9

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
