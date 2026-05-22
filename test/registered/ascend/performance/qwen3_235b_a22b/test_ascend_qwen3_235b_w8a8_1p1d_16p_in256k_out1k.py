import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    QWEN3_235B_A22B_INSTRUCT_2507_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

MODEL_CONFIG = {
    "model_path": QWEN3_235B_A22B_INSTRUCT_2507_W8A8_MODEL_PATH,
    "prefill_envs": {
        "ASCEND_USE_FIA": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "ASCEND_LAUNCH_BLOCKING": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "HCCL_BUFFSIZE": "1500",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "128",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    },
    "decode_envs": {
        "ASCEND_USE_FIA": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "SGLANG_DEEPEP_BF16_DISPATCH": "0",
        "HCCL_BUFFSIZE": "4000",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    },
    "prefill_args": [
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--quantization",
        "modelslim",
        "--chunked-prefill-size",
        "-1",
        "--skip-server-warmup",
        "--device",
        "npu",
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.45,
        "--max-running-requests",
        1,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
    ],
    "decode_args": [
        "--disaggregation-mode",
        "decode",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        0.8,
        "--disable-cuda-graph",
        "--device",
        "npu",
        "--disable-radix-cache",
        "--quantization",
        "modelslim",
        "--chunked-prefill-size",
        "8192",
        "--skip-server-warmup",
        "--tp-size",
        16,
        "--max-running-requests",
        1,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--disable-overlap-schedule",
    ],
    "router_args": [],
}


class TestQwen235bW8A8(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = AISBENCHMARK
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 25600
    output_len = 1000
    random_range_ratio = 1

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
