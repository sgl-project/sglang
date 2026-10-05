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
  - a draft whose attention backend is off the translated MHA rails
    declines: it would read the fused rows without the KV-index translator;
  - so does a draft under --dcp-size > 1, where each rank's host rows hold
    only its share of the tokens the replicated draft reads, and a draft
    under a host-pool-backed cache, which needs a device pool of its own;
  - an explicit draft KV dtype unlike the host's declines;
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
    FusedDraftDecision,
    FusedDraftPlacement,
    draft_kv_profile,
    place_fused_draft,
)
from sglang.srt.runtime_context import get_parallel, override_platform
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

    def test_swa_state_asymmetric_and_misaligned_drafts_decline(self):
        for profile in (
            _profile(swa_layer_ids=(0,)),
            _profile(num_state_layers=1),
            _profile(head_dim=64, v_head_dim=32),
            # 4 heads x 5 dims x 2 B = 40 B: not a 16-B-aligned entry part.
            _profile(head_dim=5, v_head_dim=5),
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
        cfg.use_mla_backend = False
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


class TestFusedDraftDecision(CustomTestCase):
    """The target's boot decision over a host whose full sub-pool fuses."""

    def _decide(
        self,
        *,
        algorithm="EAGLE",
        draft_kv_dtype=None,
        host_kv_dtype=_DTYPE,
        kv_cache_dtype_flag="auto",
        attention_arch=None,
        draft_backend=None,
        target_backends=("triton", "triton"),
        dcp_size=1,
        hicache=False,
        external_linker=False,
        retraction_backup="none",
    ):
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.mem_cache import kv_cache_configurator as kvc
        from sglang.srt.speculative import draft_worker_common

        cfg = kvc.KVCacheConfigurator.__new__(kvc.KVCacheConfigurator)
        cfg.is_hybrid_swa = True
        cfg.mambaish_config = None
        cfg.use_mla_backend = False
        cfg.is_draft_worker = False
        cfg.spec_algorithm = SimpleNamespace(
            is_eagle=lambda: algorithm == "EAGLE",
            is_dflash_family=lambda: algorithm in ("DFLASH", "DSPARK"),
        )
        cfg.model_config = SimpleNamespace(is_multi_layer_eagle=False)
        cfg.kv_cache_dtype = host_kv_dtype
        cfg.spec_aux_config = SimpleNamespace(
            eagle_draft_num_layers=1,
            draft_kv_num_layers=1,
            dflash_draft_num_layers=1,
            draft_model_config=SimpleNamespace(
                is_hybrid_swa=False,
                is_deepseek_v4_arch=False,
                num_nextn_predict_layers=None,
                get_num_kv_heads=lambda tp: 4,
                head_dim=64,
                v_head_dim=64,
                dtype=torch.bfloat16,
                attention_arch=attention_arch or AttentionArch.MHA,
            ),
        )
        memory = SimpleNamespace(
            enable_unified_memory=True,
            enable_hierarchical_cache=hicache,
            enable_unified_cache_external_linker=external_linker,
        )
        disagg = SimpleNamespace(
            disaggregation_decode_retraction_backup=retraction_backup
        )
        spec = SimpleNamespace(
            speculative_num_steps=1,
            speculative_draft_kv_cache_dtype=draft_kv_dtype,
            speculative_draft_attention_backend=draft_backend,
        )
        with (
            patch.object(kvc, "get_memory", return_value=memory),
            patch.object(kvc, "get_spec", return_value=spec),
            patch.object(kvc, "get_disagg", return_value=disagg),
            patch.object(
                kvc,
                "get_model",
                return_value=SimpleNamespace(kv_cache_dtype=kv_cache_dtype_flag),
            ),
            patch.object(kvc, "attention_backends", return_value=target_backends),
            patch.object(draft_worker_common, "get_spec", return_value=spec),
            patch.object(
                draft_worker_common, "attention_backends", return_value=target_backends
            ),
            patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None),
            get_parallel().override(attn_tp_size=1, attn_dcp_size=dcp_size),
            override_platform(is_xpu=False, is_hip=False),
        ):
            return cfg._fused_draft_decision()

    def test_a_stateless_draft_is_placed(self):
        self.assertIsNotNone(self._decide().placement)

    def test_a_non_mha_draft_keeps_the_private_pool(self):
        """The region holds dense MHA K/V rows. An MLA draft is kept out by its
        own architecture, not by the accident of asymmetric head dims."""
        from sglang.srt.configs.model_config import AttentionArch

        declined = self._decide(attention_arch=AttentionArch.MLA)
        self.assertIsNone(declined.placement)
        self.assertIn("MLA", declined.declined)

    def test_a_draft_kv_dtype_unlike_the_host_keeps_the_private_pool(self):
        """A fused draft stores its rows in the host's KV dtype; an explicit
        draft dtype that differs declines instead of mis-typing those rows,
        and one that matches the host's fuses."""
        self.assertIsNotNone(self._decide(draft_kv_dtype="auto").placement)
        declined = self._decide(draft_kv_dtype="fp8_e4m3")
        self.assertIsNone(declined.placement)
        self.assertIn("KV cache dtype", declined.declined)
        for algorithm in ("EAGLE", "DFLASH"):
            placed = self._decide(
                algorithm=algorithm,
                draft_kv_dtype="fp8_e4m3",
                host_kv_dtype=torch.float8_e4m3fn,
                kv_cache_dtype_flag="fp8_e4m3",
            )
            self.assertIsNotNone(placed.placement, algorithm)

    def test_a_draft_off_the_translated_rails_keeps_the_private_pool(self):
        """A fused draft reads its rows through the KV-index translator, which
        only triton, flashinfer and fa3 carry on every draft path. The draft's
        backend is the published one, a model hook's declaration included;
        otherwise it is what the draft runner inherits from the target."""
        # Kimi-Linear + DSPARK on SM100, where a model hook declares trtllm_mha.
        declined = self._decide(algorithm="DSPARK", draft_backend="trtllm_mha")
        self.assertIsNone(declined.placement)
        self.assertIn("trtllm_mha", declined.declined)
        # A DFLASH-family draft runs the target's prefill backend, or the
        # platform default when that is not a draft backend.
        for target_backends, fuses in (
            (("fa4", "fa4"), False),
            (("fa3", "flashmla"), True),
            (("flashmla", "flashmla"), True),
        ):
            decision = self._decide(algorithm="DSPARK", target_backends=target_backends)
            self.assertEqual(decision.placement is not None, fuses, target_backends)
        # An EAGLE draft with no backend of its own runs the target's pair.
        self.assertIsNone(self._decide(target_backends=("fa3", "flashmla")).placement)

    def test_dcp_keeps_the_private_pool(self):
        """Under --dcp-size > 1 each rank's host rows hold only its share of
        the widened id space, while the draft, outside the DCP group, reads
        every token."""
        for algorithm in ("DSPARK", "EAGLE"):
            declined = self._decide(algorithm=algorithm, dcp_size=2)
            self.assertIsNone(declined.placement, algorithm)
            self.assertIn("--dcp-size 2", declined.declined)

    def test_a_host_pool_backed_cache_keeps_the_private_pool(self):
        """HiCache, the external linker and host-pool retraction build the
        draft's host pool off its own device pool, which a fused draft does
        not have; DSPARK + HiCache keeps working on the private pool."""
        for kwargs, flag in (
            (dict(hicache=True), "--enable-hierarchical-cache"),
            (dict(external_linker=True), "--enable-unified-cache-external-linker"),
            (dict(retraction_backup="host_pool"), "host_pool"),
        ):
            declined = self._decide(algorithm="DSPARK", **kwargs)
            self.assertIsNone(declined.placement, flag)
            self.assertIn(flag, declined.declined)


class TestMambaHostPrivateDraftRefused(CustomTestCase):
    """A mamba host's unified buffer takes the whole KV budget, so a private
    draft pool on top of it would overcommit; the fused arm is the only
    EAGLE-family or DFLASH arm such a host builds. DSPARK keeps the private
    pool it booted with on these hosts before it could fuse."""

    _DECLINED = "the draft's K/V rows are asymmetric"

    def _resolve(self, *, decision, algorithm="EAGLE"):
        from sglang.srt.mem_cache import kv_cache_configurator as kvc
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        cfg = kvc.KVCacheConfigurator.__new__(kvc.KVCacheConfigurator)
        cfg.is_draft_worker = False
        cfg.spec_algorithm = SpeculativeAlgorithm[algorithm]
        with patch.object(
            kvc.KVCacheConfigurator, "_fused_draft_decision", return_value=decision
        ):
            return cfg._fused_draft_for_mamba_factory()

    def test_a_declined_eagle_draft_is_refused(self):
        for algorithm in ("EAGLE", "EAGLE3"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, "rows are asymmetric"):
                    self._resolve(
                        decision=FusedDraftDecision(declined=self._DECLINED),
                        algorithm=algorithm,
                    )

    def test_a_declined_dflash_draft_is_refused_and_dspark_keeps_its_pool(self):
        """DFLASH is new on the unified pool, so its decliner is refused like
        EAGLE's. DSPARK already booted here with a private pool, so its
        decliner still gets one instead of a boot failure."""
        declined = FusedDraftDecision(declined=self._DECLINED)
        with self.assertRaisesRegex(ValueError, "rows are asymmetric"):
            self._resolve(decision=declined, algorithm="DFLASH")
        self.assertIsNone(self._resolve(decision=declined, algorithm="DSPARK"))

    def test_a_placed_draft_and_other_algorithms_pass(self):
        placement = _place(_profile()).placement
        for algorithm in ("EAGLE", "DFLASH", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                self.assertIs(
                    self._resolve(
                        decision=FusedDraftDecision(placement=placement),
                        algorithm=algorithm,
                    ),
                    placement,
                )
        self.assertIsNone(
            self._resolve(decision=FusedDraftDecision(), algorithm="NONE")
        )


class TestSWAHostPrivateDraftRefused(CustomTestCase):
    """A hybrid-SWA host's boot solve prices a private EAGLE draft at the
    target's per-token size, so a draft that does not fuse is refused there
    too; the fused arm is the only EAGLE arm such a host builds."""

    def _resolve(self, *, decision, eagle=True):
        from sglang.srt.mem_cache import kv_cache_configurator as kvc

        cfg = kvc.KVCacheConfigurator.__new__(kvc.KVCacheConfigurator)
        cfg.is_draft_worker = False
        cfg.spec_algorithm = SimpleNamespace(is_eagle=lambda: eagle)
        with patch.object(
            kvc.KVCacheConfigurator, "_fused_draft_decision", return_value=decision
        ):
            return cfg._fused_draft_for_swa_factory()

    def test_a_declined_eagle_draft_is_refused(self):
        with self.assertRaisesRegex(ValueError, "KV cache dtype"):
            self._resolve(
                decision=FusedDraftDecision(
                    declined="the draft's KV cache dtype (bf16) differs from the host's"
                )
            )

    def test_a_placed_draft_and_other_algorithms_pass(self):
        placement = _place(_profile()).placement
        self.assertIs(
            self._resolve(decision=FusedDraftDecision(placement=placement)), placement
        )
        self.assertIsNone(self._resolve(decision=FusedDraftDecision(), eagle=False))


if __name__ == "__main__":
    unittest.main()
