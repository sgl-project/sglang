import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_1_W4A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_1_SINGLE_NODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "1200",
}

GLM_5_1_SINGLE_NODE_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    280000,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.75,
    "--served-model-name",
    "glm-5",
    "--cuda-graph-max-bs",
    16,
    "--max-running-requests",
    128,
    "--quantization",
    "modelslim",
    "--speculative-draft-model-quantization",
    "unquant",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUGLM5_1_W4A8_16P_In3k5_Out1k5(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for GLM-5.1-w4a8 16p single node in3k5 out1k5"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = GLM_5_1_W4A8_MODEL_PATH
    other_args = GLM_5_1_SINGLE_NODE_OTHER_ARGS
    envs = GLM_5_1_SINGLE_NODE_ENVS
    dataset_name = "random"
    max_concurrency = 64
    num_prompts = 256
    input_len = 32768
    output_len = 512
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 3000
    aisbench_repeat_rate = 0.9

    def test_npu_glm5_1_w4a8_16p_in3k5_out1k5(self):
        """Run NPU performance test for GLM-5.1-w4a8 single node"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
