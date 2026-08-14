"""Automatic PP layer partition: DP core and get_pp_indices integration."""

import unittest

from sglang.srt.distributed.pp_partition import (
    _set_auto_pp_partition,
    compute_balanced_partition,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GB = 1 << 30

# Qwen3.5-397B-flavoured reference: 60 layers, every 4th is full attention.
NUM_LAYERS = 60
FULL_IDS = list(range(3, 60, 4))  # 15 full-attention layers


def _partition(**overrides):
    kwargs = dict(
        num_layers=NUM_LAYERS,
        pp_size=8,
        full_attention_layer_ids=FULL_IDS,
        weight_bytes_per_layer=1.5 * GB,
        kv_bytes_per_token_per_full_layer=64 * 1024,  # 64 KB/token/layer
        mamba_bytes_per_slot_per_linear_layer=1 * 1024 * 1024,  # 1 MB/slot/layer
        first_stage_extra_bytes=2 * GB,
        last_stage_extra_bytes=2 * GB,
        draft_kv_bytes_per_token=0.0,
        reference_num_tokens=200_000,
        reference_num_slots=512,
    )
    kwargs.update(overrides)
    return compute_balanced_partition(**kwargs)


class TestBalancedPartition(CustomTestCase):
    def assert_valid(self, partition, pp_size):
        self.assertEqual(len(partition), pp_size)
        self.assertEqual(sum(partition), NUM_LAYERS)
        self.assertTrue(all(n >= 1 for n in partition))

    def test_uniform_costs_give_even_split(self):
        # No KV/mamba/draft asymmetry: only uniform weights, so the DP must
        # fall back to a balanced count split.
        p = _partition(
            kv_bytes_per_token_per_full_layer=0,
            mamba_bytes_per_slot_per_linear_layer=0,
            first_stage_extra_bytes=0,
            last_stage_extra_bytes=0,
        )
        self.assert_valid(p, 8)
        self.assertLessEqual(max(p) - min(p), 1)

    def test_last_stage_lightened_by_draft(self):
        with_draft = _partition(
            last_stage_extra_bytes=20 * GB, draft_kv_bytes_per_token=64 * 1024
        )
        without_draft = _partition(last_stage_extra_bytes=0)
        self.assert_valid(with_draft, 8)
        # The last stage gets fewer layers once the draft's fixed+variable
        # overhead is charged to it.
        self.assertLess(with_draft[-1], without_draft[-1])

    def test_kv_cost_concentrates_full_layers_cheaply(self):
        # When KV dominates, stages heavy in full-attention layers are
        # expensive, so the DP packs more (cheap linear) layers alongside
        # them and never exceeds the even-split max stage cost.
        p = _partition(
            weight_bytes_per_layer=0.1 * GB,
            kv_bytes_per_token_per_full_layer=512 * 1024,
            reference_num_tokens=500_000,
        )
        self.assert_valid(p, 8)

        def stage_full_count(stage):
            start = sum(p[:stage])
            return len([l for l in FULL_IDS if start <= l < start + p[stage]])

        # No stage carries more full-attention layers than the even split's max.
        self.assertLessEqual(max(stage_full_count(s) for s in range(8)), 2)

    def test_remainder_goes_to_cheap_stages(self):
        # 61 layers over 8 stages: the extra layer must not silently land on
        # the heaviest (draft-loaded) last stage.
        p = _partition(num_layers=61, last_stage_extra_bytes=20 * GB)
        self.assertEqual(sum(p), 61)
        self.assertEqual(len(p), 8)
        self.assertLessEqual(p[-1], max(p))

    def test_get_pp_indices_uses_cache(self):
        try:
            _set_auto_pp_partition([30, 30])
            self.assertEqual(get_pp_indices(60, 0, 2), (0, 30))
            self.assertEqual(get_pp_indices(60, 1, 2), (30, 60))
            # pp_size=1 (the draft worker) must not read the target's cache.
            self.assertEqual(get_pp_indices(60, 0, 1), (0, 60))
        finally:
            _set_auto_pp_partition(None)

    def test_env_var_still_wins(self):
        import os

        os.environ["SGLANG_PP_LAYER_PARTITION"] = "10,50"
        try:
            _set_auto_pp_partition([30, 30])
            self.assertEqual(get_pp_indices(60, 0, 2), (0, 10))
            self.assertEqual(get_pp_indices(60, 1, 2), (10, 60))
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]
            _set_auto_pp_partition(None)


if __name__ == "__main__":
    unittest.main()
