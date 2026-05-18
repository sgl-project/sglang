import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)

MODEL_CONFIG = {
    "model_path": DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        "HCCL_BUFFSIZE": "3500",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_BUFFSIZE": "800",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "78",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
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
        0.62,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        32,
        "--context-length",
        8192,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        20480,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        "--dp-size",
        8,
        "--enable-dp-attention",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
    ],
    "decode_args": [
        "--nnodes",
        "1",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        16,
        "--dp-size",
        16,
        "--mem-fraction-static",
        0.805,
        "--max-running-requests",
        416,
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--cuda-graph-bs",
        2,
        4,
        6,
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
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        2,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        3,
        "--prefill-round-robin-balance",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--tokenizer-worker-num",
        4,
        "--load-balance-method",
        ROUND_ROBIN,
    ],
    "router_args": [],
}


class TestDeepSeekR1W4A8(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 24
    max_concurrency = 416
    num_prompts = int(max_concurrency) * 4
    input_len = 3584
    output_len = 1536
    random_range_ratio = 1
    tpot = 41
    output_token_throughput = 6835

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
