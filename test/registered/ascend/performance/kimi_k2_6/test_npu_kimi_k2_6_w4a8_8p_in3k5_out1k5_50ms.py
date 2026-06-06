import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="full-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

KIMI_K2_6_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "HCCL_BUFFSIZE": "1200",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
}

KIMI_K2_6_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.7,
    "--max-running-requests",
    120,
    "--chunked-prefill-size",
    32768,
    "--context-length",
    8192,
    "--max-prefill-tokens",
    16384,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    120,
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--speculative-draft-model-quantization",
    "unquant",
]


class TestKimiK25W4A8(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = KIMI_K2_6_W4A8_MODEL_PATH
    other_args = KIMI_K2_6_OTHER_ARGS
    envs = KIMI_K2_6_ENVS
    backend = "sglang"
    dataset_name = "random"
    max_concurrency = 120
    num_prompts = 120
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    warmup_requests = 0
    tpot = 50
    output_token_throughput = 3452

    def test_kimi_k2_6_w4a8(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
