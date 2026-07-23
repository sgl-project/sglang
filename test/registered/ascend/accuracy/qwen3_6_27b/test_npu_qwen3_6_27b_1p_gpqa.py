import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_6_27B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_27B_64K_PREFIX_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
}

QWEN3_6_27B_64K_PREFIX_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    32768,
    "--max-prefill-tokens",
    32768,
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--trust-remote-code",
    "--max-running-requests",
    20,
    "--max-mamba-cache-size",
    120,
    "--mem-fraction-static",
    0.8,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    10,
    12,
    16,
    18,
    20,
    "--enable-prefill-delayer",
    "--prefill-delayer-queue-min-ratio",
    0.5,
    "--prefill-delayer-max-delay-ms",
    30000,
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUQwen3_6_27B_1P_In64k_Out1k_Prefix90_gpqa(TestNpuAccuracyTestCaseBase):
    model = QWEN3_6_27B_MODEL_PATH
    envs = QWEN3_6_27B_64K_PREFIX_ENVS
    other_args = QWEN3_6_27B_64K_PREFIX_OTHER_ARGS
    accuracy = 0.878
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    eval_batch_size = 64
    generation_config = {
        "max_tokens": 81920,
        "temperature": 1.0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }

    def test_gpqa(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
