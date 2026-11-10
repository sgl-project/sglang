import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_3_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

GLM_5_3_W8A8_16P_TWO_NODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

# NOTE: the GLM-5.3-w8a8 MTP layer is W8A8_DYNAMIC in the checkpoint; do NOT
# set --speculative-draft-model-quantization unquant (draft load would fail).
# The draft inherits --quantization modelslim by default.
GLM_5_3_W8A8_16P_TWO_NODE_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--attn-dp-size",
    8,
    "--chunked-prefill-size",
    65536,
    "--max-prefill-tokens",
    280000,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.76,
    "--context-length",
    135000,
    "--served-model-name",
    "glm-5.3-w8a8",
    "--cuda-graph-max-bs-decode",
    4,
    "--max-running-requests",
    32,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-shared-experts-fusion",
    "--load-balance-method",
    "round_robin",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]

GLM_5_3_W8A8_16P_TWO_NODE_MODEL_CONFIG = {
    "model_path": GLM_5_3_W8A8_MODEL_PATH,
    "other_args": GLM_5_3_W8A8_16P_TWO_NODE_OTHER_ARGS,
    "node_envs": GLM_5_3_W8A8_16P_TWO_NODE_ENVS,
}


class TestNPUGLM_5_3_W8A8_16P_GPQA(TestNpuAccuracyMultiNodePdMixTestCaseBase):
    """Test NPU accuracy for GLM-5.3-w8a8 16p two nodes on gpqa_diamond"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model_config = GLM_5_3_W8A8_16P_TWO_NODE_MODEL_CONFIG
    max_retries = 1
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    # Weekday nightly overrides this to gsm8k (baseline is provisional).
    dataset_baselines = {
        "gsm8k": 0.92,
    }
    # Strip the reasoning trace before extraction, as upstream gsm8k does.
    dataset_args = {
        "gsm8k": {"filters": {"remove_until": "</think>"}},
    }
    eval_batch_size = 32
    generation_config = {
        "max_tokens": 131072,
        "top_p": 0.95,
        "temperature": 1.0,
        "timeout": 7200,
        "retries": 2,
        "stream": True,
    }

    def test_npu_glm_5_3_w8a8_16p_gpqa(self):
        """Run NPU accuracy test for GLM-5.3-w8a8 16p two nodes on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
