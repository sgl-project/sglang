"""The ROCm DSA DCP prefill must weigh the gathered route against the owned one by wire bytes, and gather only page-whole prefixes below that crossing."""

import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_dcp import (
    dcp_gathered_prefill_crossover,
    dcp_use_gathered_prefill,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# GLM-5.2 MLA geometry with the bf16 partials and the fp8 KV cache.
_GLM52 = dict(
    v_head_dim=256,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    partial_dtype=torch.bfloat16,
    kv_dtype=torch.float8_e4m3fn,
)


class TestDCPGatheredPrefillCrossover(CustomTestCase):
    def test_glm52_routes(self):
        for project_before_merge, q_all_gather, want in (
            (True, False, 0.896),
            (True, True, 1.896),
            (False, False, 1.785),
            (False, True, 2.785),
        ):
            with self.subTest(
                project_before_merge=project_before_merge, q_all_gather=q_all_gather
            ):
                got = dcp_gathered_prefill_crossover(
                    **_GLM52,
                    project_before_merge=project_before_merge,
                    q_all_gather=q_all_gather,
                )
                self.assertAlmostEqual(got, want, places=3)


class TestDCPUseGatheredPrefill(CustomTestCase):
    def test_prefix_not_page_whole(self):
        self.assertFalse(dcp_use_gathered_prefill([6], 100, 16, 100.0, dcp_size=4))

    def test_one_prefix_not_page_whole(self):
        self.assertFalse(dcp_use_gathered_prefill([8, 6], 100, 16, 100.0, dcp_size=4))

    def test_zero_crossover_never_gathers(self):
        self.assertFalse(dcp_use_gathered_prefill([0], 100, 16, 0.0, dcp_size=4))
        self.assertFalse(dcp_use_gathered_prefill([8], 100, 16, 0.0, dcp_size=4))

    def test_zero_prefix_gathers(self):
        self.assertTrue(dcp_use_gathered_prefill([0], 100, 16, 0.896, dcp_size=4))

    def test_crossing_is_exclusive(self):
        # 16 heads * 100 extend tokens * crossover 1.0 = 1600 prefix tokens.
        self.assertTrue(dcp_use_gathered_prefill([1596], 100, 16, 1.0, dcp_size=4))
        self.assertFalse(dcp_use_gathered_prefill([1600], 100, 16, 1.0, dcp_size=4))


if __name__ == "__main__":
    unittest.main()
