"""With --enable-hierarchical-cache, the routed experts of a prefix that comes
back from the host tier must equal the rows of the cold run: prefix rows are
reused on a cache hit, not recomputed."""

import random
import unittest

import numpy as np
import requests

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")

_NUM_LAYERS = 48
_TOPK = 8
_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo".split()


def _long_prompt(seed: int, num_words: int = 2500) -> str:
    rng = random.Random(seed)
    words = " ".join(rng.choice(_WORDS) for _ in range(num_words))
    return f"Document {seed}: {words}\nSummary:"


class TestReturnRoutedExpertsHiCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-return-routed-experts",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                4,
                "--max-total-tokens",
                16384,
                "--page-size",
                1,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, text: str) -> dict:
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "return_routed_experts": True,
            },
            timeout=300,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _rows(self, out: dict) -> np.ndarray:
        return extract_routed_experts_from_meta_info(out).reshape(
            -1, _NUM_LAYERS, _TOPK
        )

    def test_host_tier_hit_returns_cold_rows(self):
        cold = self._generate(_long_prompt(seed=0))
        # Fill the device pool so the first prompt's prefix moves to the host tier.
        for seed in range(1, 9):
            self._generate(_long_prompt(seed=seed))
        warm = self._generate(_long_prompt(seed=0))

        cached = warm["meta_info"]["cached_tokens"]
        num_prefix_rows = min(cached, cold["meta_info"]["prompt_tokens"] - 1)
        self.assertGreater(num_prefix_rows, 0, "expected a prefix cache hit")
        cold_rows = self._rows(cold)[:num_prefix_rows]
        warm_rows = self._rows(warm)[:num_prefix_rows]
        mismatched = int((cold_rows != warm_rows).any(axis=(1, 2)).sum())
        self.assertEqual(
            mismatched,
            0,
            f"{mismatched} of {num_prefix_rows} cached prefix rows differ from the cold run",
        )


if __name__ == "__main__":
    unittest.main()
