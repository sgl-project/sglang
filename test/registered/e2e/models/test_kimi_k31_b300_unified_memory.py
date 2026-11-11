"""Nightly coverage of Kimi-K3.1 sparse attention on the unified memory pool:
FP8 latent rows and the indexer keys share each page of the full sub-pool."""

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

MODEL_PATH = "moonshotai/Kimi-K3.1-Pro"
MODEL_LOADER_EXTRA_CONFIG = '{"enable_multithread_load": true, "num_threads": 12}'
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


class TestKimiK31B300UnifiedMemory(GSM8KMixin, CustomTestCase):
    """TP8 with --enable-unified-memory and the default flashmla_kv DSA backends."""

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
