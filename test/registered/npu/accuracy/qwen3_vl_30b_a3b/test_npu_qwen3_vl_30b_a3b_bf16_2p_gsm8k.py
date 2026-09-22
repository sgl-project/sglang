import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_VL_30B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4800, suite="full-acc-4-npu-a3", nightly=True)

QWEN3_VL_30B_A3B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_VL_30B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--dtype",
    "bfloat16",
    "--device",
    "npu",
    "--mm-attention-backend",
    "ascend_attn",
    "--enable-multimodal",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    102400,
    "--max-running-requests",
    512,
    "--tp-size",
    4,
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.78,
    "--sampling-backend",
    "ascend",
]


class TestNPUQwen3_VL_30B_A3B_GSM8K(TestNpuAccuracyTestCaseBase):
    model = QWEN3_VL_30B_MODEL_PATH
    envs = QWEN3_VL_30B_A3B_ENVS
    other_args = QWEN3_VL_30B_A3B_OTHER_ARGS
    accuracy = 0.9538
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 40000,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    eval_batch_size = 64

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
