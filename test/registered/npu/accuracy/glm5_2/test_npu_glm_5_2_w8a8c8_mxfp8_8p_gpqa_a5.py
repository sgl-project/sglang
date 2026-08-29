import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_2_W8A8C8_MXFP8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-acc-8-npu-a5",
    nightly=True,
)

GLM_5_2_W4A8_8P_A5_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "HCCL_BUFFSIZE": "1000",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
}

GLM_5_2_W4A8_8P_A5_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--nnodes",
    1,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--disable-shared-experts-fusion",
    "--chunked-prefill-size",
    32768,
    "--max-prefill-tokens",
    32768,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--served-model-name",
    "GLM-5.2-w4a8",
    "--cuda-graph-bs",
    4,
    "--max-running-requests",
    16,
    "--quantization",
    "modelslim",
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
    "--speculative-draft-model-quantization",
    "unquant",
]


class TestNPUGLM_5_2_W8A8C8_MXFP8_8P_GPQA_A5(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for GLM-5___2-W8A8C8-mxfp8 8p A5 GPQA."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = GLM_5_2_W8A8C8_MXFP8_MODEL_PATH
    other_args = GLM_5_2_W4A8_8P_A5_OTHER_ARGS
    envs = GLM_5_2_W4A8_8P_A5_ENVS
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 32
    stream = True
    seed = 1

    def test_npu_glm_5_2_w8a8c8_mxfp8_8p_gpqa_a5(self):
        """Run NPU accuracy test for GLM-5___2-W8A8C8-mxfp8 8p A5 GPQA."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
