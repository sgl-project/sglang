import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_480B_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)

MODEL_CONFIG = {
    "model_path": QWEN3_480B_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_FUSED_MOE_MODE": "2",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "327680",
        "HCCL_BUFFSIZE": "1550",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "65536",
        "HCCL_BUFFSIZE": "600",
        "SGLANG_NPU_PROFILING": "0",
        "SGLANG_NPU_PROFILING_BS": "136",
        "SGLANG_NPU_FUSED_MOE_MODE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "prefill_args": [
        "--disaggregation-mode",
        "prefill",
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--tp-size",
        "16",
        "--mem-fraction-static",
        "0.7",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
        "--max-running-requests",
        "16",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "20480",
        "--max-prefill-tokens",
        "20480",
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--ep-dispatch-algorithm",
        "static",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--dtype",
        "bfloat16",
        "--disable-overlap-schedule",
    ],
    "decode_args": [
        "--disaggregation-mode",
        "decode",
        "--trust-remote-code",
        "--nnodes",
        "2",
        "--tp-size",
        "32",
        "--dp-size",
        "4",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        "544",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--ep-dispatch-algorithm",
        "static",
        "--cuda-graph-bs",
        "16",
        "32",
        "56",
        "72",
        "80",
        "88",
        "96",
        "104",
        "112",
        "120",
        "128",
        "136",
        "--disaggregation-transfer-backend",
        "ascend",
        "--watchdog-timeout",
        "9000",
        "--context-length",
        "8192",
        "--tokenizer-worker-num",
        "4",
        "--prefill-round-robin-balance",
        "--dtype",
        "bfloat16",
        "--load-balance-method",
        "round_robin",
    ],
    "router_args": [],
}


class TestQwen480bW8a8(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 680
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50.1
    output_token_throughput = 9831

    def test_throughput(self):
        hotmap_file_prefill = (
            "480_3.5k_prefill.pt"
            if self.benchmark_tool == AISBENCHMARK
            else "480_3.5k_prefill_ais.pt"
        )
        self.model_config["prefill_args"].extend(
            [
                "--init-expert-location",
                f"/root/.cache/modelscope/hub/models/hot_map/{hotmap_file_prefill}",
            ]
        )
        hotmap_file_decode = (
            "480_3.5k_decode.pt"
            if self.benchmark_tool == AISBENCHMARK
            else "480_3.5k_decode_ais.pt"
        )
        self.model_config["decode_args"].extend(
            [
                "--init-expert-location",
                f"/root/.cache/modelscope/hub/models/hot_map/{hotmap_file_decode}",
            ]
        )
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
