"""B300 per-commit CI coverage for the Kimi-K3.1 Balanced serving recipe.

Runs TP8/DCP8/EP2 with DFLASH linear ReplaySSM speculation and hierarchical
cache on eight B300 GPUs, gated on full-set GSM8K accuracy.
"""

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

register_cuda_ci(est_time=1800, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3.1-Pro"
DFLASH_DRAFT_MODEL = "RadixArk/Kimi-K3.1-Dflash2-Preview"
MODEL_LOADER_EXTRA_CONFIG = '{"enable_multithread_load": true}'
SERVER_ENV = {
    "SGLANG_DCP_PUSH_COMBINE_MAX_ROWS": "1536",
    "SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB": "32768",
}
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


class TestKimiK31B300(GSM8KMixin, CustomTestCase):
    """TP8/DCP8/EP2 recipe with DFLASH speculation and hierarchical cache."""

    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.935
    gsm8k_num_examples = 1319  # full GSM8K test split
    gsm8k_num_threads = 64  # matches --max-running-requests
    gsm8k_thinking = True
    gsm8k_temperature = 1.0
    gsm8k_top_p = 0.95
    gsm8k_max_tokens = 32768

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
                "--dcp-comm-backend",
                "fi_a2a",
                "--ep-size",
                "2",
                "--mem-fraction-static",
                "0.85",
                "--max-running-requests",
                "64",
                "--weight-loader-prefetch-checkpoints",
                "--model-loader-extra-config",
                MODEL_LOADER_EXTRA_CONFIG,
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--mamba-full-memory-ratio",
                "3.63",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DFLASH_DRAFT_MODEL,
                "--speculative-dflash-block-size",
                "8",
                "--speculative-draft-kv-cache-dtype",
                "fp8_e4m3",
                "--speculative-draft-attention-backend",
                "cutedsl_mla",
                "--enable-linear-replayssm-spec",
                "--enable-hierarchical-cache",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--mamba-max-states-per-path",
                "1",
            ],
            env=SERVER_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
            _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
