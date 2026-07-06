import os
import unittest

from sglang.srt.utils import find_local_repo_dir, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"


def _checkpoints_available(*model_paths: str) -> bool:
    for path in model_paths:
        if os.path.isdir(path):
            continue
        try:
            snapshot_dir = find_local_repo_dir(path, revision=None)
        except Exception:
            return False
        if not snapshot_dir or not os.path.isdir(snapshot_dir):
            return False
    return True


class TestBasicSanityDSpark(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    GSM8KMixin,
    CustomTestCase,
):
    served_model_name = TARGET_MODEL
    model = TARGET_MODEL

    fwd_occupancy_threshold = 60
    fwd_occupancy_max_new_tokens = 4096
    fwd_occupancy_acc_length_threshold: float = 2.0

    gsm8k_num_questions = 200
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 2.0

    attention_backend = "trtllm_mha"
    draft_attention_backend = "fa4"

    @classmethod
    def setUpClass(cls):
        if not _checkpoints_available(TARGET_MODEL, DRAFT_MODEL):
            raise unittest.SkipTest(
                f"Checkpoint(s) unavailable (gated/missing/offline): "
                f"{TARGET_MODEL}, {DRAFT_MODEL}."
            )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                cls.attention_backend,
                "--speculative-draft-attention-backend",
                cls.draft_attention_backend,
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--page-size",
                "1",
                "--enable-metrics",
                "--disable-piecewise-cuda-graph",
            ],
            env={
                "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
