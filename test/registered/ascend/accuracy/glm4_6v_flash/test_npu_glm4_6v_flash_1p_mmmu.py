import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import GLM_4_6V_FLASH_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_SET_CPU_AFFINITY": "1",
}

OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    2,
    "--chunked-prefill-size",
    16384,
    "--max-prefill-tokens",
    150000,
    "--dtype",
    "bfloat16",
    "--max-running-requests",
    32,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.75,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    32,
    "--watchdog-timeout",
    9000,
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm45",
]


class TestQwen3(TestNpuAccuracyTestCaseBase):
    model = GLM_4_6V_FLASH_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.711
    datasets = ["mmmu"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_mmmu(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
