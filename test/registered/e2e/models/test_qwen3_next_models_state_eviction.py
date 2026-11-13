import re
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=350, stage="weekly", runner_config="4-gpu-h100")

# The eviction policies act only on a full cache: with the eager extra buffer
# and checkpoint level 3 every request caches its tracked states, so pools this
# small make GSM8K evict.
_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--attention-backend",
    "triton",
    "--mamba-track-interval",
    "64",
    "--hybrid-cache-checkpoints-level",
    "3",
    "--page-size",
    "1",
    "--max-total-tokens",
    "65536",
    "--max-mamba-cache-size",
    "256",
    "--enable-metrics",
]
_EVICTED = re.compile(r"^sglang:evicted_tokens_total\S* [1-9]", re.MULTILINE)


class TestQwen3NextMambaUnitEviction(GSM8KMixin, DefaultServerBase):
    model = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.90
    gsm8k_max_tokens = 1024
    other_args = [*_ARGS, "--enable-mamba-unit-eviction"]

    def test_gsm8k(self):
        super().test_gsm8k()
        metrics = requests.get(f"{self.base_url}/metrics", timeout=30).text
        self.assertRegex(metrics, _EVICTED, "nothing was evicted")


class TestQwen3NextGreedyDualSizeEviction(GSM8KMixin, DefaultServerBase):
    model = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.90
    gsm8k_max_tokens = 1024
    other_args = [*_ARGS, "--radix-eviction-policy", "gds"]

    def test_gsm8k(self):
        super().test_gsm8k()
        metrics = requests.get(f"{self.base_url}/metrics", timeout=30).text
        self.assertRegex(metrics, _EVICTED, "nothing was evicted")


if __name__ == "__main__":
    unittest.main()
