# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`place_fused_draft`: which host/draft pairs fuse, and where each runner's
layers land.

Every decline is the private-pool fallback, so a wrong answer never fails
the boot -- it silently changes what the draft binds and what the boot solve
prices. Pinned:
  - a replicated head under multi-layer EAGLE gets one lane RANGE per runner
    (one shared region would let the runners clobber each other's KV);
  - a per-depth head serves one depth per runner and needs one runner per
    depth;
  - a draft with SWA or recurrent-state layers of its own, or asymmetric K/V
    rows, declines: the fused arm binds one dense pool over the host's full
    slots;
  - the profile divides the draft's heads by attn_tp, as the target does;
  - a placement whose runner lane counts do not fill its region is refused;
  - the priced entry counts the layers THIS runner owns, not the whole model's.

    python -m pytest test/registered/unit/mem_cache/test_fused_draft_placement.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftKVGeometry,
    DraftKVProfile,
    FusedDraftPlacement,
    draft_kv_profile,
    place_fused_draft,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DTYPE = torch.bfloat16


def _profile(
    *,
    num_layers=1,
    swa_layer_ids=(),
    num_depths=1,
    num_state_layers=0,
    head_dim=64,
    v_head_dim=64,
):
    return DraftKVProfile(
        num_layers=num_layers,
        full=DraftKVGeometry(head_num=4, head_dim=head_dim, v_head_dim=v_head_dim),
        swa_layer_ids=swa_layer_ids,
        num_depths=num_depths,
        num_state_layers=num_state_layers,
    )


def _place(profile, num_runners=1):
    return place_fused_draft(
        profile=profile,
        num_runners=num_runners,
        store_dtype=_DTYPE,
    )


class TestPlaceFusedDraft(CustomTestCase):
    def test_replicated_head_gets_one_slot_range_per_runner(self):
        decision = _place(_profile(num_layers=2), num_runners=3)
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.region.lane_num, 6)
        for r in range(3):
            self.assertEqual(placement.lanes_for(r), range(2 * r, 2 * r + 2))

    def test_per_depth_head_serves_one_depth_per_runner(self):
        decision = _place(_profile(num_layers=8, num_depths=8), num_runners=2)
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.region.lane_num, 2)
        self.assertEqual(placement.lanes_for(1), range(1, 2))

    def test_per_depth_head_needs_one_runner_per_depth(self):
        self.assertIsNone(_place(_profile(num_layers=8, num_depths=8), 1).placement)
        self.assertIsNone(_place(_profile(num_layers=8, num_depths=8), 9).placement)

    def test_swa_state_and_asymmetric_drafts_decline(self):
        for profile in (
            _profile(swa_layer_ids=(0,)),
            _profile(num_state_layers=1),
            _profile(head_dim=64, v_head_dim=32),
        ):
            decision = _place(profile)
            self.assertIsNone(decision.placement)
            self.assertIsNotNone(decision.declined)

    def test_region_carries_the_profile_geometry(self):
        placement = _place(_profile()).placement
        self.assertEqual(
            placement.region,
            DenseDraftRegion(
                lane_num=1, head_num=4, head_dim=64, v_head_dim=64, store_dtype=_DTYPE
            ),
        )


class TestDraftKVProfile(CustomTestCase):
    def test_heads_are_divided_by_attn_tp(self):
        mc = SimpleNamespace(
            is_hybrid_swa=True,
            is_deepseek_v4_arch=False,
            swa_attention_layer_ids=[0],
            num_nextn_predict_layers=None,
            get_num_kv_heads=lambda tp: max(1, 8 // tp),
            head_dim=64,
            v_head_dim=32,
        )
        with patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None):
            profile = draft_kv_profile(mc, num_layers=1, attn_tp_size=2)
        self.assertEqual(
            profile.full, DraftKVGeometry(head_num=4, head_dim=64, v_head_dim=32)
        )
        self.assertEqual(profile.swa_layer_ids, (0,))
        self.assertEqual(profile.num_depths, 1)
        self.assertEqual(profile.num_state_layers, 0)

    def _linear_trunk_mc(self, hf_text_config):
        return SimpleNamespace(
            is_hybrid_swa=False,
            is_deepseek_v4_arch=False,
            num_nextn_predict_layers=1,
            get_num_kv_heads=lambda tp: 8,
            head_dim=64,
            v_head_dim=64,
            hf_text_config=hf_text_config,
        )

    def test_a_nextn_head_of_a_linear_trunk_owns_no_state(self):
        """BUG REGRESSION. A NEXTN head ships inside the trunk checkpoint and
        inherits its config CLASS, so `mamba2_cache_params` lists the TRUNK's
        state layers while the head is a full-attention block. Counting them
        declined a fusable draft to its private pool."""
        mc = self._linear_trunk_mc(SimpleNamespace())
        trunk = SimpleNamespace(mamba2_cache_params=SimpleNamespace(layers=[0, 1, 2]))
        with patch(
            "sglang.srt.configs.hybrid_arch.mambaish_config", return_value=trunk
        ):
            profile = draft_kv_profile(mc, num_layers=1, attn_tp_size=1)
        self.assertEqual(profile.num_state_layers, 0)

    def test_a_conv_chain_head_still_counts_its_state(self):
        """The discriminator must not silence a head that really owns state:
        only a conv-chain MTP config declares `mtp_local_layer_ids`."""
        mc = self._linear_trunk_mc(SimpleNamespace(mtp_local_layer_ids=[0]))
        trunk = SimpleNamespace(mamba2_cache_params=SimpleNamespace(layers=[0, 1, 2]))
        with patch(
            "sglang.srt.configs.hybrid_arch.mambaish_config", return_value=trunk
        ):
            profile = draft_kv_profile(mc, num_layers=1, attn_tp_size=1)
        self.assertEqual(profile.num_state_layers, 3)


class TestFusedDraftPlacement(CustomTestCase):
    def test_runner_lane_counts_must_fill_the_region(self):
        region = DenseDraftRegion(
            lane_num=2, head_num=1, head_dim=8, store_dtype=_DTYPE
        )
        with self.assertRaises(AssertionError):
            FusedDraftPlacement(region=region, runner_lane_counts=(1,))
        with self.assertRaises(AssertionError):
            FusedDraftPlacement(region=region, runner_lane_counts=())


class TestFusedEntryPricing(CustomTestCase):
    """The boot solve prices a fused entry through the same spec the pool
    factory builds, so the two cannot drift. The factory builds the full
    sub-pool from this runner's OWN layer slice; pricing from the whole-model
    split over-counts every layer another pipeline rank holds, and the solve
    then hands out fewer tokens than fit."""

    def _configurator(self, *, whole, hybrid_swa, owned=(), span=(0, 0)):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        cfg.is_hybrid_swa = hybrid_swa
        cfg.layer_info = SimpleNamespace(
            full_attention_layer_ids=list(owned),
            start_layer=span[0],
            end_layer=span[1],
        )
        cfg.mambaish_config = SimpleNamespace(full_attention_layer_ids=whole)
        cfg.model_config = SimpleNamespace(
            full_attention_layer_ids=whole,
            head_dim=8,
            get_num_kv_heads=lambda *_: 1,
        )
        cfg.kv_cache_dtype = _DTYPE
        return cfg

    def _priced(self, cfg):
        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=_DTYPE
        )
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            return cfg._full_host_spec(region).layer_num

    def test_an_swa_host_prices_this_runners_slice(self):
        cfg = self._configurator(
            whole=[0, 1, 2, 3], hybrid_swa=True, owned=[0, 1], span=(0, 2)
        )
        self.assertEqual(self._priced(cfg), 2)

    def test_a_mamba_host_prices_only_the_layers_in_its_span(self):
        cfg = self._configurator(whole=[0, 1, 2, 3], hybrid_swa=False, span=(2, 4))
        self.assertEqual(self._priced(cfg), 2)


if __name__ == "__main__":
    unittest.main()
