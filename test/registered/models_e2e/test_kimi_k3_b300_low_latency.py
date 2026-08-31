"""B300 per-commit CI coverage for the Kimi-K3 Low Latency recipe.

Runs the TP8 DSPARK recipe on eight B300 GPUs and checks MMMU-Pro quality,
speculative acceptance, and single-request decode performance.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MMMUProMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=1415, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3"
DSPARK_DRAFT_MODEL = "RadixArk/Kimi-K3-DSpark"
MODEL_LOADER_EXTRA_CONFIG = '{"enable_multithread_load": true, "num_threads": 12}'
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


def _stop_server(process):
    if process:
        kill_process_tree(process.pid)
        _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


class TestKimiK3B300LowLatency(MMMUProMixin, SpecDecodingMixin, CustomTestCase):
    """TP8 Low Latency recipe with DSPARK linear ReplaySSM speculation."""

    mmmu_pro_score_threshold = 0.75
    mmmu_pro_num_examples = 200
    mmmu_pro_load_preset_from_model_id = MODEL_PATH
    # MMMU-Pro's long multimodal reasoning has a lower speculative average than
    # GSM8K (2.62 in the first B300 run). Keep a workload-specific regression
    # gate here; test_bs_1_speed below retains the stricter single-prompt gate.
    mmmu_pro_accept_length_thres = 2.4
    # Both scale with how far that one greedy prompt runs, and speed is
    # end-to-end, so launch and TTFT are amortized over the output -- it sits
    # well below the steady decode rate the server logs. Coarse guards only.
    accept_length_thres = 4.0
    bs_1_speed_thres = 300

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                MODEL_LOADER_EXTRA_CONFIG,
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--mamba-full-memory-ratio",
                "0.86",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DSPARK_DRAFT_MODEL,
                "--speculative-dspark-block-size",
                "7",
                "--enable-linear-replayssm-spec",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


if __name__ == "__main__":
    unittest.main()
