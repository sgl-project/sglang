import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_OMNI_30B_A3B_THINKING_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "HCCL_BUFFSIZE": "400",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--max-running-requests",
    162,
    "--disable-radix-cache",
    # "--speculative-draft-model-quantization",
    # "unquant",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    35000,
    # "--speculative-algorithm",
    # "EAGLE3",
    # "--speculative-draft-model-path",
    # QWEN3_A3B_EAGLE_MODEL_PATH,
    # "--speculative-num-steps",
    # 3,
    # "--speculative-eagle-topk",
    # 1,
    # "--speculative-num-draft-tokens",
    # 4,
    "--tp-size",
    2,
    "--mem-fraction-static",
    0.87,
    "--cuda-graph-bs",
    1,
    5,
    15,
    40,
    70,
    100,
    120,
    130,
    140,
    146,
    150,
    154,
    156,
    158,
    160,
    162,
    "--dtype",
    "bfloat16",
]


class TestQwen3(TestNpuAccuracyTestCaseBase):
    model = QWEN3_OMNI_30B_A3B_THINKING_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.576
    datasets = ["mmmu"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_mmmu(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
