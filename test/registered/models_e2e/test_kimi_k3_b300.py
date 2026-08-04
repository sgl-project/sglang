"""B300 per-commit CI coverage for Kimi-K3 serving recipes.

Runs the Low Latency DSPARK recipe and the Balanced DCP/HiCache recipe on
eight B300 GPUs. Each server must preserve basic model quality on GSM8K.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = (
    "/data/radixark/model-cache/hub/models--moonshotai--Kimi-K3/"
    "snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DSPARK_DRAFT_MODEL = "RadixArk/Kimi-K3-DSpark"
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


def _stop_server(process):
    if process:
        kill_process_tree(process.pid)
        _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


class TestKimiK3B300LowLatency(GSM8KMixin, CustomTestCase):
    """TP8 Low Latency recipe with DSPARK linear ReplaySSM speculation."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200

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
                "--weight-loader-prefetch-checkpoints",
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
                "--weight-loader-prefetch-checkpoints",
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


if __name__ == "__main__":
    unittest.main()
