import os
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=7200,
    suite="nightly-acc-16-npu-a3",
    nightly=True,
)

MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "140000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "1024",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
}

MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_OTHER_ARGS = [
    "--tp-size",
    8,
    "--mem-fraction-static",
    0.63,
    "--max-running-requests",
    26,
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
    "--enable-prefill-delayer",
    "--prefill-max-requests",
    10,
    "--chunked-prefill-size",
    67072,
    "--max-prefill-token",
    67000,
    "--cuda-graph-bs",
    2,
    4,
    8,
    12,
    16,
    18,
    20,
    22,
    24,
    26,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--fuseep-mode",
    2,
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
]


class TestNPUMiniMaxM2_5_W8A8_4P_Gpqa(TestNpuAccuracyTestCaseBase):
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_OTHER_ARGS
    envs = MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_ENVS
    accuracy = 0.852
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_accuracy(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
