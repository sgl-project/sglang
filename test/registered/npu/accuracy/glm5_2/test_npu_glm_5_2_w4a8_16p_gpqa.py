import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_2_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

GLM_5_2_W4A8_16P_TWO_NODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "TRANSFORMERS_VERBOSITY": "error",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_W4A8_16P_TWO_NODE_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--nnodes",
    2,
    # "--dp-size",
    # 8,
    # "--enable-dp-attention",
    "--chunked-prefill-size",
    65536,
    "--max-prefill-tokens",
    280000,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.70,
    "--served-model-name",
    "glm-5",
    "--cuda-graph-max-bs",
    32,
    "--max-running-requests",
    32,
    "--quantization",
    "modelslim",
    # "--speculative-draft-model-quantization",
    # "unquant",
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
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
]

GLM_5_2_W4A8_16P_TWO_NODE_MODEL_CONFIG = {
    "model_path": GLM_5_2_W4A8_MODEL_PATH,
    "other_args": GLM_5_2_W4A8_16P_TWO_NODE_OTHER_ARGS,
    "node_envs": GLM_5_2_W4A8_16P_TWO_NODE_ENVS,
}


class TestNPUGLM_5_2_W4A8_16P_GPQA(TestNpuAccuracyMultiNodePdMixTestCaseBase):
    """Test NPU accuracy for GLM-5.2-w4a8 16p two nodes on gpqa_diamond"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model_config = GLM_5_2_W4A8_16P_TWO_NODE_MODEL_CONFIG
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    eval_batch_size = 32
    generation_config = {"max_tokens": 65536, "temperature": 1.0}

    def test_npu_glm_5_2_w4a8_16p_gpqa(self):
        """Run NPU accuracy test for GLM-5.2-w4a8 16p two nodes on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
