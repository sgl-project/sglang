import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_32B_EAGLE_MODEL_PATH,
    QWEN3_32B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_32B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
}

QWEN3_32B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    64,
    "--disable-radix-cache",
    "--speculative-draft-model-quantization",
    "unquant",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    65536,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_32B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.72,
    "--cuda-graph-bs",
    64,
    "--dtype",
    "bfloat16",
]


class TestQwen32B_GPQA(TestNpuAccuracyTestCaseBase):
    model = QWEN3_32B_MODEL_PATH
    envs = QWEN3_32B_ENVS
    other_args = QWEN3_32B_OTHER_ARGS
    accuracy = 0.516
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    eval_batch_size = 64
    generation_config = {"max_tokens": 40000, "temperature": 1.0}

    def test_accuracy(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
