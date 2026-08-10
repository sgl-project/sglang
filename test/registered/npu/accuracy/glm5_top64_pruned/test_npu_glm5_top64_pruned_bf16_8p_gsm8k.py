import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM5_TOP64_PRUNED_GSM8K_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3600, suite="base-c-test-acc-16-npu-a3")

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_BUFFSIZE": "1000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}
OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--tp-size",
    "16",
    "--chunked-prefill-size",
    "16384",
    "--trust-remote-code",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.7",
    "--served-model-name",
    "glm-5",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    16,
]


class TestNPUGLM5_Top64_Pruned_GSM8K(TestNpuAccuracyTestCaseBase):

    model = GLM5_TOP64_PRUNED_GSM8K_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.48
    datasets = ["gsm8k"]
    generation_config = {
        "max_tokens": 2048,
        "temperature": 0.01,
    }
    eval_batch_size = 16
    limit = 100

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
