"""Stage-a basic sanity with EAGLE3 spec decoding enabled. Mirrors
test_basic_sanity.py with the spec-decoding path active."""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=200, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=200, suite="stage-a-test-1-gpu-small-amd")

# --- KV_SIZE_THRES begin (auto; update_memory_thresholds.py) ---
# gpu=5090 updated=2026-07-18
KV_SIZE_THRES = 142.0
# --- KV_SIZE_THRES end ---


class TestBasicSanityEagle3(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    GSM8KMixin,
    CustomTestCase,
):
    served_model_name = DEFAULT_TARGET_MODEL_EAGLE3
    # CUDA 5090 + Llama-3.1-8B measured ~99 median in CI with async-assert
    # probes off in base-a. AMD EAGLE3 currently sustains lower single-batch
    # occupancy and needs a longer measurement window to avoid too few
    # non-NaN samples.
    fwd_occupancy_threshold = 80.0 if is_in_amd_ci() else 98.0
    fwd_occupancy_max_new_tokens = 4096 if is_in_amd_ci() else 2048
    fwd_occupancy_acc_length_threshold: float = 1.6

    model = DEFAULT_TARGET_MODEL_EAGLE3
    gsm8k_num_questions = 1400
    gsm8k_accuracy_thres = 0.74

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_TARGET_MODEL_EAGLE3,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                # Canonical EAGLE3 sglang config: fp16 + triton attention.
                # bf16 + flashinfer cutlass RMSNorm hits a SM120 dtype
                # mismatch on the draft model's input_layernorm.
                "--dtype",
                "float16",
                "--attention-backend",
                "triton",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_DRAFT_MODEL_EAGLE3,
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--cuda-graph-max-bs-decode",
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
