"""Stage-a basic sanity with DFLASH spec decoding enabled. Mirrors
test_basic_sanity.py / test_basic_sanity_eagle3.py with the DFLASH path active
(overlap scheduling on by default)."""

import unittest

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=200, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=200, suite="stage-a-test-1-gpu-small-amd")


class TestBasicSanityDFlash(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    GSM8KMixin,
    CustomTestCase,
):
    served_model_name = DEFAULT_TARGET_MODEL_DFLASH
    fwd_occupancy_threshold = 80.0 if is_in_amd_ci() else 95
    fwd_occupancy_max_new_tokens = 4096 if is_in_amd_ci() else 2048
    # DFLASH accepts a full block per verify, so its acc length runs well above
    # EAGLE3's; keep a safe lower bound here.
    fwd_occupancy_acc_length_threshold: float = 2.0

    model = DEFAULT_TARGET_MODEL_DFLASH
    gsm8k_num_questions = 1400
    gsm8k_accuracy_thres = 0.74
    gsm8k_accept_length_thres = 2.8

    attention_backend = "triton" if is_hip() else "trtllm_mha"
    draft_attention_backend = "triton"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_TARGET_MODEL_DFLASH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                cls.attention_backend,
                "--speculative-draft-attention-backend",
                cls.draft_attention_backend,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DEFAULT_DRAFT_MODEL_DFLASH,
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--enable-metrics",
                "--disable-piecewise-cuda-graph",
            ],
            env={"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
