import unittest

import requests

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_xpu_ci(est_time=360, suite="extra-test-1-gpu-xpu")

PROMPT = list(range(1000, 1384))
FOLLOW_UP = list(range(2000, 2032))
XPU_ARGS = ["--device", "xpu", "--attention-backend", "intel_xpu"]


class MambaRadixCacheXPUMixin:
    expected_page_size = 128
    expected_strategy = "extra_buffer"

    def _generate(self, input_ids):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 16, "temperature": 0},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def _flush(self):
        requests.post(self.base_url + "/flush_cache", timeout=30).raise_for_status()

    def test_radix_cache_layout(self):
        info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertFalse(info["disable_radix_cache"])
        self.assertEqual(info["page_size"], self.expected_page_size)
        self.assertEqual(info["mamba_radix_cache_strategy"], self.expected_strategy)

    def test_follow_up_turn_hits_cache_and_matches_cold_output(self):
        self._flush()
        first = self._generate(PROMPT)
        follow_up = PROMPT + first["output_ids"] + FOLLOW_UP
        self._flush()
        cold = self._generate(follow_up)
        self._flush()
        self._generate(PROMPT)
        warm = self._generate(follow_up)
        self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
        self.assertEqual(warm["output_ids"], cold["output_ids"])


class TestQwen35RadixCacheNoOverlapXPU(MambaRadixCacheXPUMixin, DefaultServerBase):
    model = "Qwen/Qwen3.5-0.8B"
    other_args = XPU_ARGS + ["--disable-overlap-schedule"]


class TestQwen35RadixCacheOverlapXPU(MambaRadixCacheXPUMixin, DefaultServerBase):
    model = "Qwen/Qwen3.5-0.8B"
    other_args = XPU_ARGS


class TestLfm2RadixCacheNoBufferXPU(MambaRadixCacheXPUMixin, DefaultServerBase):
    model = "LiquidAI/LFM2-350M"
    other_args = XPU_ARGS
    expected_page_size = 1
    expected_strategy = "no_buffer"


if __name__ == "__main__":
    unittest.main()
