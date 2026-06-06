import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_6_35B_A3B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-2-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_35B_A3B_3K5_1K5_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "100",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "ASCEND_USE_FIA": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
}

QWEN3_6_35B_A3B_3K5_1K5_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    131072,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests",
    4,
    "--max-mamba-cache-size",
    4,
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs",
    1,
    2,
    3,
    4,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
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


class TestNPUQwen3_6_35BA3B_1P_In3k5_Out1k5_aime26(TestAscendAccuracyTestCaseBase):
    model = QWEN3_6_35B_A3B_MODEL_PATH
    envs = QWEN3_6_35B_A3B_3K5_1K5_ENVS
    other_args = QWEN3_6_35B_A3B_3K5_1K5_OTHER_ARGS
    accuracy = 0.927
    datasets = ["aime26"]
    few_shot_num = 0
    eval_batch_size = 4
    generation_config = {
        "max_tokens": 131072,
        "temperature": 0.2,
        "repetition_penalty": 1.08,
    }

    def test_aime26(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
