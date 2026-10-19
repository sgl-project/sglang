"""B200 NVFP4 E2E coverage for NVIDIA Nemotron 3.5 Lightning.

The three cases exercise the production NVFP4 checkpoint without speculation,
with DFlash, and with DSpark. MTP is already covered by the Nemotron model
family tests; the external-draft paths are the new coverage in this file.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=863, stage="extra-b", runner_config="4-gpu-b200")

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
DFLASH_DRAFT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash"
DSPARK_DRAFT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"

SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_SCORE_THRESHOLD = 0.80

BASE_ARGS = [
    "--mamba-backend",
    "flashinfer",
    "--mamba-ssm-dtype",
    "float16",
    "--enable-mamba-cache-stochastic-rounding",
    "--mamba-cache-philox-rounds",
    "5",
    "--mem-fraction-static",
    "0.85",
    "--cuda-graph-max-bs-decode",
    "16",
    "--reasoning-parser",
    "nemotron_3",
    "--tool-call-parser",
    "qwen3_coder",
]


class _Nemotron35LightningServer:
    speculative_args: list[str] = []
    model = try_cached_model(MODEL)
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = GSM8K_SCORE_THRESHOLD

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS + cls.speculative_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestNvidiaNemotron35LightningNVFP4(
    _Nemotron35LightningServer, GSM8KMixin, CustomTestCase
):
    """Normal autoregressive serving."""


class TestNvidiaNemotron35LightningNVFP4DFlash(
    _Nemotron35LightningServer, GSM8KMixin, CustomTestCase
):
    """DFlash with the published W4A16 draft checkpoint."""

    speculative_args = [
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DFLASH_DRAFT_MODEL,
        "--speculative-dflash-block-size",
        "6",
    ]


class TestNvidiaNemotron35LightningNVFP4DSpark(
    _Nemotron35LightningServer, GSM8KMixin, CustomTestCase
):
    """DSpark with the published bonus-anchor W4A16 draft checkpoint."""

    speculative_args = [
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        DSPARK_DRAFT_MODEL,
        "--speculative-dspark-block-size",
        "3",
    ]


if __name__ == "__main__":
    unittest.main()
