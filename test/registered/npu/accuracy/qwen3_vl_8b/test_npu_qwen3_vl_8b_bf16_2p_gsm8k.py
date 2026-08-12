import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_VL_8B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3600, suite="base-c-test-acc-4-npu-a3")

QWEN3_VL_8B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_VL_8B_OTHER_ARGS = [
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.8,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--sampling-backend",
    "ascend",
    "--tool-call-parser",
    "qwen",
    "--reasoning-parser",
    "qwen3",
    "--cuda-graph-bs",
    8,
    16,
    32,
    64,
    128,
    256,
    "--dtype",
    "bfloat16",
]


class TestNPUQwen3_VL_8B_GSM8K(TestNpuAccuracyTestCaseBase):
    model = QWEN3_VL_8B_MODEL_PATH
    envs = QWEN3_VL_8B_ENVS
    other_args = QWEN3_VL_8B_OTHER_ARGS
    accuracy = 0.9553
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "repetition_penalty": 1.0,
        "presence_penalty": 2.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    eval_batch_size = 64

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
