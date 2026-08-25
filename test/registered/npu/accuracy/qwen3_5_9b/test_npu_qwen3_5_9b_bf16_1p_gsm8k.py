import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_5_9B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3600, suite="base-c-test-acc-2-npu-a3")
register_npu_ci(est_time=2800, suite="nightly-acc-2-npu-a3", nightly=True)

QWEN3_5_9B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_5_9B_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--enable-dp-attention",
    "--chunked-prefill-size",
    4096,
    "--max-prefill-tokens",
    280000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs",
    16,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
]


class TestNPUQwen3_5_9B_GSM8K(TestNpuAccuracyTestCaseBase):
    model = QWEN3_5_9B_MODEL_PATH
    envs = QWEN3_5_9B_ENVS
    other_args = QWEN3_5_9B_OTHER_ARGS
    accuracy = 0.8350
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 8192,
        "temperature": 0.6,
    }
    eval_batch_size = 64
    limit = 100

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
