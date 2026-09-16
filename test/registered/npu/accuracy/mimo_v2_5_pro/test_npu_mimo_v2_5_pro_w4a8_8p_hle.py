import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="nightly-acc-8-npu-a5-test",
    nightly=True,
)

# TODO: Add MIMO_V2_5_PRO_FP4_MODEL_PATH to test_npu_performance_utils.py and test_ascend_utils.py
MIMO_V2_5_PRO_FP4_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/<ORG>/MiMo-V2.5-Pro-FP4-DFlash"
)
# TODO: Add MIMO_V2_5_PRO_DFLASH_MODEL_PATH to test_npu_performance_utils.py and test_ascend_utils.py
MIMO_V2_5_PRO_DFLASH_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/<ORG>/MiMo-V2.5-Pro-FP4-DFlash/dflash"
)
# TODO: Update HLE dataset path for CI environment
HLE_DATASET_PATH = "/mnt/share/w00937173/run_file/mimo-v2.5-pro/data/hle_dataset"

MIMO_V2_5_PRO_FP4_4P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "300",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "20",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "0",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
}

MIMO_V2_5_PRO_FP4_4P_OTHER_ARGS = [
    "--served-model-name",
    MIMO_V2_5_PRO_FP4_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--mem-fraction-static",
    0.905,
    "--tp-size",
    8,
    "--nnodes",
    1,
    "--chunked-prefill-size",
    8192,
    "--max-total-tokens",
    600000,
    "--max-running-requests",
    8,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    6,
    8,
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-model-path",
    MIMO_V2_5_PRO_DFLASH_MODEL_PATH,
    "--speculative-num-draft-tokens",
    8,
    "--enable-metrics",
]


class TestNPUMiMoV2_5_Pro_W4A8_4P_HLE(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for MiMo-V2.5-Pro-FP4 4p single node on HLE"""

    model = MIMO_V2_5_PRO_FP4_MODEL_PATH
    other_args = MIMO_V2_5_PRO_FP4_4P_OTHER_ARGS
    envs = MIMO_V2_5_PRO_FP4_4P_ENVS
    accuracy = 0.33
    datasets = ["hle"]
    dataset_args = {
        "hle": {
            "local_path": HLE_DATASET_PATH,
            "extra_params": {
                "include_multi_modal": False,
            },
        }
    }
    few_shot_num = 0
    eval_batch_size = 5
    limit = 3
    generation_config = {
        "temperature": 0,
        "parallel_tool_calls": True,
    }
    stream = True
    judge_model_args = {
        "model_id": "MiMo-V2.5-Pro-FP4-DFlash",
    }

    def test_hle(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
