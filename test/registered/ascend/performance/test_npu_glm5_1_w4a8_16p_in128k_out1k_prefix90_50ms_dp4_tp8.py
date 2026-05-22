import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_1_W4A8_MODEL_PATH,
    TestAscendPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_1_TWO_NODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2000",
}

GLM_5_1_TWO_NODE_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--chunked-prefill-size",
    65536,
    "--max-prefill-tokens",
    280000,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.65,
    "--served-model-name",
    "glm-5",
    "--cuda-graph-max-bs",
    8,
    "--max-running-requests",
    32,
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

GLM_5_1_TWO_NODE_MODEL_CONFIG = {
    "model_path": GLM_5_1_W4A8_MODEL_PATH,
    "other_args": GLM_5_1_TWO_NODE_OTHER_ARGS,
    "node_envs": GLM_5_1_TWO_NODE_ENVS,
}


class TestNPUGLM5_1_W4A8_32P_In3k5_Out1k5(TestAscendPerfMultiNodePdMixTestCaseBase):
    """Test NPU performance for GLM-5.1-w4a8 32p two nodes in3k5 out1k5"""

    model_config = GLM_5_1_TWO_NODE_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 4
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 3000
    aisbench_repeat_rate = 0.9

    def test_npu_glm5_1_w4a8_32p_in3k5_out1k5(self):
        """Run NPU performance test for GLM-5.1-w4a8 two nodes"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
