import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MOONLIGHT_16B_A3B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

MODEL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

MODEL_OTHER_ARGS = [
    "--tp-size",
    2,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--dtype",
    "bfloat16",
    "--mem-fraction-static",
    0.8,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    4096,
    "--max-prefill-tokens",
    16384,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    "--max-running-requests",
    128,
    "--watchdog-timeout",
    9000,
]


class TestNPUMoonlight16B_A3B_GSM8K(TestNpuAccuracyTestCaseBase):
    model = MOONLIGHT_16B_A3B_MODEL_PATH
    envs = MODEL_ENVS
    other_args = MODEL_OTHER_ARGS
    accuracy = 0.8370
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {"max_tokens": 7168, "temperature": 1.0}
    eval_batch_size = 64

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
