import re
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=350, stage="weekly", runner_config="4-gpu-h100")

_COMMON_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--attention-backend",
    "triton",
]
_EVICTED = re.compile(r"^sglang:evicted_tokens_total\S* [1-9]", re.MULTILINE)


class TestQwen3NextCacheMissAttribution(
    GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST
    cache_chunk_size = 64
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.90
    gsm8k_max_tokens = 1024
    other_args = [
        *_COMMON_ARGS,
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--mamba-track-interval",
        "2",
        "--page-size",
        "1",
        "--enable-cache-miss-attribution",
    ]


# Page size 2: a request that ends inside a page leaves a leaf owning that
# page, which the next turn copies; states tracked only every 64 tokens, and
# checkpoint level 3 caches each of them, so that GSM8K fills the small pools.
class TestQwen3NextMambaExactCheckpoint(GSM8KMixin, DefaultServerBase):
    model = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.90
    gsm8k_max_tokens = 1024
    other_args = [
        *_COMMON_ARGS,
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "64",
        "--hybrid-cache-checkpoints-level",
        "3",
        "--page-size",
        "2",
        "--max-total-tokens",
        "65536",
        "--max-mamba-cache-size",
        "256",
        "--enable-metrics",
        "--enable-mamba-exact-checkpoint",
    ]

    def _generate(self, input_ids, max_new_tokens):
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        ).json()

    def test_gsm8k(self):
        super().test_gsm8k()
        metrics = requests.get(f"{self.base_url}/metrics", timeout=30).text
        self.assertRegex(metrics, _EVICTED, "nothing was evicted")

    def test_the_next_turn_resumes_at_the_exact_end(self):
        """The first turn processes 107 tokens (108 if an overlapped step
        feeds the last one), ending inside a page and off the tracked grid;
        the next turn resumes after all of them."""
        requests.get(self.base_url + "/flush_cache")
        first = list(range(1000, 1100))
        output = self._generate(first, max_new_tokens=8)["output_ids"]
        second = first + output + [2000, 2001, 2002]
        cached = self._generate(second, max_new_tokens=4)["meta_info"]["cached_tokens"]
        self.assertIn(cached, (len(first) + 7, len(first) + 8))


if __name__ == "__main__":
    unittest.main()
