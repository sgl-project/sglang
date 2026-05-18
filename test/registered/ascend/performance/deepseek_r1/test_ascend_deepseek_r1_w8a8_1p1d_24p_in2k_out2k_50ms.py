import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)

MODEL_CONFIG = {
    "model_path": DEEPSEEK_R1_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "HCCL_BUFFSIZE": "1600",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "SGLANG_USE_AG_AFTER_QLORA": "1",
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_BUFFSIZE": "800",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "102",
        "TASK_QUEUE_ENABLE": "1",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "SGLANG_NPU_FUSED_MOE_MODE": "1",
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
        0.8,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        20,
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
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        "--dp-size",
        4,
        "--enable-dp-attention",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--enable-attn-tp-input-scattered",
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
        0.81,
        "--max-running-requests",
        1088,
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--moe-dense-tp",
        "1",
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
        28,
        30,
        32,
        34,
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
        "--tokenizer-worker-num",
        4,
        "--prefill-round-robin-balance",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--load-balance-method",
        ROUND_ROBIN,
    ],
    "router_args": [],
}


class TestDeepSeekR1W8A8(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 24
    max_concurrency = 1088
    num_prompts = 12800
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    tpot = 46
    output_token_throughput = 21429

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
