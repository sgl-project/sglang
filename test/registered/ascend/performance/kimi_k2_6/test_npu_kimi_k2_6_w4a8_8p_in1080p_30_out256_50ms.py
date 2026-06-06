import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    BENCHMARK_TOOL_DEFAULT,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="full-8-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

KIMI_K2_6_IN1080P_30_OUT256_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "1800",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

KIMI_K2_6_IN1080P_30_OUT256_OTHER_ARGS = [
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.7,
    "--max-running-requests",
    80,
    "--chunked-prefill-size",
    -1,
    "--context-length",
    8192,
    "--prefill-max-requests",
    1,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--cuda-graph-bs",
    1,
    2,
    4,
    6,
    8,
    10,
    "--disable-radix-cache",
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


class TestNPUKimiK2_6_W4A8_8P_IN1080P_30_OUT256_50ms(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Kimi-K2.6-w4a8 8p multimodal in1080p+30 out256"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    model = KIMI_K2_6_W4A8_MODEL_PATH
    other_args = KIMI_K2_6_IN1080P_30_OUT256_OTHER_ARGS
    envs = KIMI_K2_6_IN1080P_30_OUT256_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    image_resolution = "1920x1080"
    image_count = 1
    max_concurrency = 20
    num_prompts = 20
    request_rate = 1
    input_len = 30
    output_len = 256
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 1374

    def test_npu_kimi_k2_6_w4a8_8p_in1080p_30_out256_50ms(self):
        """Run NPU performance test for Kimi-K2.6-w4a8 multimodal in1080p+30 out256"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
