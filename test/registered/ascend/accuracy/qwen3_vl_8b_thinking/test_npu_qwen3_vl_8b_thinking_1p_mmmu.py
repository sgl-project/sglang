import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_VL_8B_THINKING_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
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
    "--max-running-requests",
    16,
    "--max-prefill-tokens",
    16384,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--tp-size",
    2,
    "--mem-fraction-static",
    0.894,
    "--cuda-graph-bs",
    1,
    5,
    15,
    16,
    "--dtype",
    "bfloat16",
    # "--speculative-draft-model-quantization",
    # "unquant",
    # "--speculative-algorithm",
    # "EAGLE3",
    # "--speculative-draft-model-path",
    # QWEN3_8B_EAGLE_MODEL_PATH,
    # "--speculative-num-steps",
    # 4,
    # "--speculative-eagle-topk",
    # 1,
    # "--speculative-num-draft-tokens",
    # 5,
]


class TestQwen3(TestAscendAccuracyTestCaseBase):
    model = QWEN3_VL_8B_THINKING_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.741
    datasets = ["mmmu"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 16

    def test_mmmu(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
