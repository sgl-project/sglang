"""Nightly coverage of the KDA speculative ReplaySSM ring under unified memory,
where the ring lives in the arena's request sub-pool and the CuTe DSL verify
writes it by request row."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=900, stage="nightly", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3"
DSPARK_DRAFT_MODEL = "RadixArk/Kimi-K3-DSpark"
MODEL_LOADER_EXTRA_CONFIG = '{"enable_multithread_load": true, "num_threads": 12}'
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


class TestKimiK3B300DSparkReplaySSMUnifiedMemory(GSM8KMixin, CustomTestCase):
    """TP8 DSPARK speculation with the ReplaySSM spec ring under
    --enable-unified-memory."""

    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 64

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
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DSPARK_DRAFT_MODEL,
                "--speculative-dspark-block-size",
                "7",
                "--enable-linear-replayssm-spec",
                "--enable-unified-memory",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process:
            terminate_and_kill_process_tree(process)
            _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
