"""B300 per-commit CI coverage for Kimi-K3 serving recipes.

Runs the Low Latency DSPARK, Balanced DCP/HiCache, and MegaMoE recipes on
eight B300 GPUs. Each server must preserve basic model quality on GSM8K, and
the Low Latency recipe must also preserve single-request decode performance.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=1200, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3"
DSPARK_DRAFT_MODEL = "RadixArk/Kimi-K3-DSpark"
MEGAMOE_URL = "http://0.0.0.0:30000"
MODEL_LOADER_EXTRA_CONFIG = '{"enable_multithread_load": true, "num_threads": 12}'
MEGAMOE_ENV = {
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
}
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


def _stop_server(process):
    if process:
        kill_process_tree(process.pid)
        _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


class TestKimiK3B300LowLatency(GSM8KMixin, SpecDecodingMixin, CustomTestCase):
    """TP8 Low Latency recipe with DSPARK linear ReplaySSM speculation."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 37
    # Gated on GSM8K rather than on test_bs_1_speed below: a 200-question
    # average holds steady when a numerics change moves where the single
    # greedy prompt hits EOS.
    gsm8k_accept_length_thres = 4.5
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


class TestKimiK3B300Balanced(GSM8KMixin, CustomTestCase):
    """TP8/DCP8 Balanced recipe with hierarchical cache."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 98

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
                "--dcp-size",
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
                "7.21",
                "--enable-hierarchical-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


class TestKimiK3B300MegaMoE(GSM8KMixin, CustomTestCase):
    """TP8/EP8/DCP8 MegaMoE recipe with DSPARK speculation."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 22

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = MEGAMOE_URL
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--moe-a2a-backend",
                "megamoe",
                "--ep",
                "8",
                "--dcp-size",
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
                "5.13",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DSPARK_DRAFT_MODEL,
                "--speculative-dspark-block-size",
                "7",
                "--enable-linear-replayssm-spec",
            ],
            env=MEGAMOE_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


if __name__ == "__main__":
    unittest.main()
