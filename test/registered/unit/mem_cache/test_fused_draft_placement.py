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
"""`place_fused_draft` and the target's fused-draft decision: which drafts
fuse into the host's entries, and where each runner's layers land.

A decline keeps a draft on a private pool (or refuses it at boot); admitting
one that cannot read the fused rows correctly gives silently wrong drafts.
Pinned:
  - a replicated head under multi-layer EAGLE gets one lane RANGE per runner
    (one shared region would let the runners clobber each other's KV);
  - a per-depth head serves one depth per runner and needs one runner per
    depth;
  - a draft with a layer kind no registered host kind serves declines, and
    so does a draft under --dcp-size > 1 (each rank's rows hold only its
    share of the tokens), an explicit draft KV dtype unlike the host's, and
    a draft under HiCache or host-pool decode retraction (they build the
    draft's host pool off a device pool of its own);
  - no attention backend takes part: a fused draft reads its rows through
    the same views and id spaces as the target, whatever backend it runs;
  - the profile divides the draft's heads by attn_tp, as the target does,
    and owns recurrent state only for a per-depth conv-chain head;
  - the registry refuses a duplicate host or a second primary host for one
    layer kind and reports hosts in registration order, every placed host
    reports itself in the boot log, and runner ranges must tile a region.

    python -m pytest test/registered/unit/mem_cache/test_fused_draft_placement.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.layout.fused_draft import (
    FULL_HOST,
    HOST_KINDS,
    LAYER_FULL,
    LAYER_STATE,
    LAYER_WINDOW,
    DenseDraftRegion,
    DenseHostKind,
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftDecision,
    FusedDraftPlacement,
    PlacementContext,
    RunnerLanes,
    draft_kv_profile,
    place_fused_draft,
    register_host_kind,
)
from sglang.srt.runtime_context import get_parallel, override_platform
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_HOSTS = ("full", "swa")
_HOSTS_TRI = ("full", "swa", "mamba")
_DTYPE = torch.bfloat16
_WINDOW = DraftKVGeometry(head_num=2, head_dim=64, v_head_dim=64)
_STATE = DraftStateGeometry(
    conv_state_shapes=((2, 4), (2, 4)),
    conv_dtype=torch.bfloat16,
    temporal_state_shape=(1, 8, 8),
    temporal_dtype=torch.float32,
)


def _profile(
    *,
    num_layers=1,
    window_layer_ids=(),
    num_depths=1,
    state_layer_ids=(),
    head_dim=64,
    v_head_dim=64,
    window=64,
):
    kinds = {}
    full_ids = tuple(i for i in range(num_layers) if i not in window_layer_ids)
    if full_ids:
        kinds[LAYER_FULL] = DraftLayerSet(
            layer_ids=full_ids,
            geometry=DraftKVGeometry(
                head_num=4, head_dim=head_dim, v_head_dim=v_head_dim
            ),
        )
    if window_layer_ids:
        kinds[LAYER_WINDOW] = DraftLayerSet(
            layer_ids=tuple(window_layer_ids), geometry=_WINDOW, window=window
        )
    if state_layer_ids:
        kinds[LAYER_STATE] = DraftLayerSet(
            layer_ids=tuple(state_layer_ids), geometry=_STATE
        )
    return DraftKVProfile(num_layers=num_layers, num_depths=num_depths, kinds=kinds)


def _place(profile, num_runners=1, host_names=_HOSTS, target_window=None):
    return place_fused_draft(
        profile=profile,
        num_runners=num_runners,
        ctx=PlacementContext(host_names=host_names, target_window=target_window),
        store_dtype=_DTYPE,
    )


def _dense_only_registry():
    return patch.dict(HOST_KINDS, {"full": FULL_HOST}, clear=True)


class TestPlaceFusedDraft(CustomTestCase):
    def test_replicated_head_gets_one_slot_range_per_runner(self):
        decision = _place(_profile(num_layers=2), num_runners=3)
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.region("full").lane_num, 6)
        self.assertEqual(placement.hosts(), ("full",))
        for r in range(3):
            self.assertEqual(placement.lanes_for(r, "full"), range(2 * r, 2 * r + 2))
            self.assertEqual(placement.lanes_for(r, "swa"), range(0))

    def test_per_depth_head_serves_one_depth_per_runner(self):
        decision = _place(_profile(num_layers=8, num_depths=8), num_runners=2)
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.region("full").lane_num, 2)
        self.assertEqual(placement.lanes_for(1, "full"), range(1, 2))

    def test_a_tri_pool_host_fuses_a_full_only_draft(self):
        decision = _place(_profile(), host_names=("full", "swa", "mamba"))
        self.assertIsNotNone(decision.placement, decision.declined)
        self.assertEqual(decision.placement.hosts(), ("full",))

    def test_window_layers_ride_in_the_swa_sub_pool_within_the_target_window(self):
        # MiMoV2MTP shape: one window layer per runner, the target's own window.
        decision = _place(
            _profile(window_layer_ids=(0,), window=128),
            num_runners=3,
            target_window=128,
        )
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertIsNone(decision.note)
        self.assertEqual(placement.hosts(), ("swa",))
        self.assertIsNone(placement.region("full"))
        self.assertEqual(placement.region("swa").lane_num, 3)
        self.assertEqual(placement.region("swa").head_num, _WINDOW.head_num)
        for r in range(3):
            self.assertEqual(placement.lanes_for(r, "swa"), range(r, r + 1))
            self.assertEqual(placement.lanes_for(r, "full"), range(0))

    def test_per_depth_head_places_each_depth_by_its_kind(self):
        # Inkling shape: depth 1 is a local (window) block, depths 0 and 2 full.
        decision = _place(
            _profile(num_layers=8, num_depths=8, window_layer_ids=(1,), window=64),
            num_runners=3,
            target_window=128,
        )
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.region("full").lane_num, 2)
        self.assertEqual(placement.region("swa").lane_num, 1)
        self.assertEqual(placement.lanes_for(0, "full"), range(0, 1))
        self.assertEqual(placement.lanes_for(1, "swa"), range(0, 1))
        self.assertEqual(placement.lanes_for(1, "full"), range(0))
        self.assertEqual(placement.lanes_for(2, "full"), range(1, 2))

    def test_window_layers_fall_back_to_the_full_sub_pool(self):
        """A window wider than the target's, an undeclared window, or a host
        without an swa sub-pool sends the window layers to the full sub-pool
        with their own row geometry; the decision says why."""
        for kwargs in (
            dict(window=256, target_window=128),
            dict(window=None, target_window=128),
            dict(window=64, target_window=None),
            dict(window=64, target_window=128, host_names=("full", "mamba")),
        ):
            profile = _profile(window_layer_ids=(0,), window=kwargs.pop("window"))
            decision = _place(profile, num_runners=2, **kwargs)
            placement = decision.placement
            self.assertIsNotNone(placement, decision.declined)
            self.assertIn("ride in the 'full' sub-pool", decision.note)
            self.assertIsNone(placement.region("swa"))
            self.assertEqual(placement.region("full").lane_num, 2)
            self.assertEqual(placement.region("full").head_num, _WINDOW.head_num)
            self.assertEqual(placement.lanes_for(1, "full"), range(1, 2))

    def test_a_fold_cannot_mix_two_row_geometries(self):
        """One region holds one row geometry: window layers the swa sub-pool
        turns away fold into the full sub-pool only when their rows match the
        full layers', else the whole draft declines."""
        decision = _place(
            _profile(num_layers=2, window_layer_ids=(1,), window=256),
            target_window=128,
        )
        self.assertIsNone(decision.placement)
        self.assertIn("cannot share", decision.declined)

    def test_per_depth_head_needs_one_runner_per_depth(self):
        self.assertIsNone(_place(_profile(num_layers=8, num_depths=8), 1).placement)
        self.assertIsNone(_place(_profile(num_layers=8, num_depths=8), 9).placement)

    def test_state_layers_ride_in_the_mamba_sub_pool(self):
        # Inkling shape: one block per depth, every depth carrying conv state.
        decision = _place(
            _profile(num_layers=8, num_depths=8, state_layer_ids=tuple(range(8))),
            num_runners=3,
            host_names=_HOSTS_TRI,
        )
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.hosts(), ("full", "mamba"))
        self.assertEqual(placement.region("mamba").lane_num, 3)
        self.assertEqual(placement.region("mamba").state, _STATE)
        self.assertEqual(
            placement.region("mamba").entry_bytes(), 3 * _STATE.layer_bytes()
        )
        for r in range(3):
            self.assertEqual(placement.lanes_for(r, "mamba"), range(r, r + 1))
        # A replicated head carries every state layer per runner.
        placement = _place(
            _profile(num_layers=2, state_layer_ids=(0, 1)),
            num_runners=2,
            host_names=_HOSTS_TRI,
        ).placement
        self.assertEqual(placement.region("mamba").lane_num, 4)
        self.assertEqual(placement.lanes_for(1, "mamba"), range(2, 4))

    def test_state_layers_decline_without_a_state_host(self):
        decision = _place(_profile(state_layer_ids=(0,)), host_names=_HOSTS)
        self.assertIsNone(decision.placement)
        self.assertIn("'mamba'", decision.declined)

    def test_a_layer_kind_with_no_host_kind_declines(self):
        with _dense_only_registry():
            for profile, noun in (
                (_profile(num_layers=2, window_layer_ids=(0,)), "sliding-window"),
                (_profile(state_layer_ids=(0,)), "recurrent-state"),
            ):
                decision = _place(profile)
                self.assertIsNone(decision.placement)
                self.assertIn(f"1 {noun} layer(s)", decision.declined)

    def test_short_rows_reserve_padding_between_parts(self):
        region = _place(_profile(head_dim=5, v_head_dim=5)).placement.region("full")
        self.assertEqual(region.entry_bytes(), 96)
        k, v = region.parts(64)
        self.assertEqual((k.row_bytes(), v.row_bytes()), (40, 40))
        self.assertEqual((k.offset_bytes, v.offset_bytes), (64, 112))

    def test_asymmetric_rows_fuse_with_their_v_width(self):
        placement = _place(_profile(head_dim=64, v_head_dim=32)).placement
        self.assertIsNotNone(placement)
        self.assertEqual(placement.region("full").resolved_v_head_dim(), 32)
        self.assertEqual(placement.region("full").entry_bytes(), 4 * (64 + 32) * 2)

    def test_short_v_rows_have_their_own_padding(self):
        region = _place(_profile(head_dim=64, v_head_dim=5)).placement.region("full")
        self.assertEqual(region.entry_bytes(), 512 + 48)

    def test_region_carries_the_profile_geometry(self):
        placement = _place(_profile()).placement
        self.assertEqual(
            placement.region("full"),
            DenseDraftRegion(
                lane_num=1, head_num=4, head_dim=64, v_head_dim=64, store_dtype=_DTYPE
            ),
        )


def _config(**overrides):
    fields = dict(
        is_hybrid_swa=True,
        is_deepseek_v4_arch=False,
        swa_attention_layer_ids=[0],
        num_nextn_predict_layers=None,
        hf_text_config=SimpleNamespace(),
        get_num_kv_heads=lambda tp: max(1, 8 // tp),
        head_dim=64,
        v_head_dim=32,
        get_swa_num_kv_heads=lambda tp: max(1, 4 // tp),
        swa_head_dim=64,
        swa_v_head_dim=64,
        sliding_window_size=128,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


_TRUNK_STATE = SimpleNamespace(
    mamba2_cache_params=SimpleNamespace(
        shape=SimpleNamespace(conv=((2, 4), (2, 4)), temporal=(1, 8, 8)),
        dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
        layers=list(range(36)),
    )
)


class TestDraftKVProfile(CustomTestCase):
    def test_heads_are_divided_by_attn_tp(self):
        with patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None):
            profile = draft_kv_profile(_config(), num_layers=2, attn_tp_size=2)
        self.assertEqual(profile.num_depths, 1)
        self.assertEqual(set(profile.kinds), {LAYER_FULL, LAYER_WINDOW})
        full = profile.kinds[LAYER_FULL]
        self.assertEqual(full.layer_ids, (1,))
        self.assertEqual(
            full.geometry, DraftKVGeometry(head_num=4, head_dim=64, v_head_dim=32)
        )
        window = profile.kinds[LAYER_WINDOW]
        self.assertEqual(window.layer_ids, (0,))
        self.assertEqual(
            window.geometry, DraftKVGeometry(head_num=2, head_dim=64, v_head_dim=64)
        )
        self.assertEqual(window.window, 128)

    def test_a_nextn_head_of_a_linear_attention_trunk_owns_no_state(self):
        """The head's config is the trunk's, so it is mamba-ish by class and
        lists the trunk's state layers, while the head is a full-attention
        block: profiling those layers would decline a draft that fuses."""
        config = _config(
            is_hybrid_swa=False, swa_attention_layer_ids=[], num_nextn_predict_layers=1
        )
        with patch(
            "sglang.srt.configs.hybrid_arch.mambaish_config", return_value=_TRUNK_STATE
        ):
            profile = draft_kv_profile(config, num_layers=1, attn_tp_size=1)
        self.assertEqual(set(profile.kinds), {LAYER_FULL})

    def test_a_per_depth_conv_chain_head_owns_one_state_block_per_depth(self):
        config = _config(
            swa_attention_layer_ids=[1],
            num_nextn_predict_layers=2,
            hf_text_config=SimpleNamespace(mtp_local_layer_ids=[1]),
        )
        with patch(
            "sglang.srt.configs.hybrid_arch.mambaish_config", return_value=_TRUNK_STATE
        ):
            profile = draft_kv_profile(config, num_layers=2, attn_tp_size=1)
        self.assertEqual(profile.num_depths, 2)
        self.assertEqual(profile.kinds[LAYER_FULL].layer_ids, (0,))
        self.assertEqual(profile.kinds[LAYER_WINDOW].layer_ids, (1,))
        state = profile.kinds[LAYER_STATE]
        self.assertEqual(state.layer_ids, (0, 1))
        self.assertEqual(state.geometry, _STATE)


class TestHostKindRegistry(CustomTestCase):
    """Registration is the extension point: a silent overwrite or a second
    primary host would re-route every draft of that kind."""

    def _dense(self, **overrides):
        fields = dict(name="aux", serves=LAYER_WINDOW, family="dense")
        fields.update(overrides)
        return DenseHostKind(**fields)

    def test_a_duplicate_host_name_is_refused(self):
        with _dense_only_registry(), self.assertRaises(AssertionError):
            register_host_kind(self._dense(name="full"))

    def test_one_primary_host_per_layer_kind(self):
        with _dense_only_registry(), self.assertRaises(AssertionError):
            register_host_kind(self._dense(serves=LAYER_FULL))

    def test_a_fallback_host_must_be_registered_first(self):
        with _dense_only_registry(), self.assertRaises(AssertionError):
            register_host_kind(self._dense(fallback_host="ghost"))

    def test_hosts_follow_registration_order(self):
        """Binders and the boot log walk `hosts()`; dict order of a placement's
        regions must not leak into it."""
        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=_DTYPE
        )
        with _dense_only_registry():
            register_host_kind(self._dense())
            placement = FusedDraftPlacement.from_counts(
                counts={"aux": [1], "full": [1]},
                regions={"aux": region, "full": region},
            )
            self.assertEqual(placement.hosts(), ("full", "aux"))

    def test_describe_names_the_dense_geometry(self):
        """Log tooling keys on this line; it is part of the boot contract."""
        region = DenseDraftRegion(
            lane_num=2, head_num=1, head_dim=128, v_head_dim=64, store_dtype=_DTYPE
        )
        self.assertEqual(
            FULL_HOST.describe(region=region, lanes=[(0,), (), (1,)]),
            "fused draft region in 'full': 2 lane(s) x 1 kv head(s) x 128/64 k/v "
            "head_dim @ torch.bfloat16 = 768 B/token; runner lanes [(0,), (), (1,)]",
        )


class TestFusedDraftPlacement(CustomTestCase):
    def test_runner_ranges_must_tile_the_region(self):
        region = DenseDraftRegion(
            lane_num=2, head_num=1, head_dim=8, store_dtype=_DTYPE
        )
        with self.assertRaises(AssertionError):
            FusedDraftPlacement(
                runners=(RunnerLanes(ranges={"full": (0, 1)}),),
                regions={"full": region},
            )
        with self.assertRaises(AssertionError):
            FusedDraftPlacement(runners=(RunnerLanes(ranges={"full": (0, 1)}),))
        with self.assertRaises(AssertionError):
            FusedDraftPlacement.from_counts(counts={"full": [1]}, regions={})

    def test_a_region_needs_a_registered_host(self):
        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=_DTYPE
        )
        with self.assertRaises(AssertionError):
            FusedDraftPlacement.from_counts(
                counts={"ghost": [1]}, regions={"ghost": region}
            )


class TestBootLogReportsEveryPlacedHost(CustomTestCase):
    """Every placed host reports through its kind's `describe`, so a host kind
    that forgets one dies at boot instead of hiding a placement, and each kind
    names its OWN geometry: a shared line would drop to the two fields the
    kinds have in common and stop saying what was placed."""

    def _resolve(self, placement):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        with self.assertLogs(
            "sglang.srt.mem_cache.kv_cache_configurator", level="INFO"
        ) as captured:
            out = cfg._fused_draft_for_pool_factory(
                FusedDraftDecision(placement=placement)
            )
        self.assertIs(out, placement)
        return "\n".join(captured.output)

    def test_dense_only_placement_reports_its_region(self):
        joined = self._resolve(
            FusedDraftPlacement.from_counts(
                counts={"full": [1]},
                regions={
                    "full": DenseDraftRegion(
                        lane_num=1, head_num=2, head_dim=16, store_dtype=_DTYPE
                    )
                },
            )
        )
        self.assertIn("fused draft region in 'full'", joined)
        self.assertIn("runner lanes [(0,)]", joined)
        self.assertNotIn("fused draft state", joined)

    def test_mixed_placement_reports_each_kind_with_its_own_geometry(self):
        joined = self._resolve(
            FusedDraftPlacement.from_counts(
                counts={"full": [1], "mamba": [1]},
                regions={
                    "full": DenseDraftRegion(
                        lane_num=1, head_num=2, head_dim=16, store_dtype=_DTYPE
                    ),
                    "mamba": DraftStateRegion(lane_num=1, state=_STATE),
                },
            )
        )
        self.assertIn("fused draft region in 'full'", joined)
        self.assertIn(
            "fused draft state in 'mamba': 1 lane(s) x 2 conv stream(s)", joined
        )


def _fused_plan(cfg, placement, host):
    """``host``'s spec in the plan the pool is built from, and the fused entry
    the configurator prices for it."""
    from sglang.srt.mem_cache.kv.plan import plan_kv_storage

    # A slotted dataclass: the decision is patched on the class.
    with patch.object(
        kcc.KVCacheConfigurator,
        "_fused_draft_decision",
        return_value=FusedDraftDecision(placement=placement),
    ):
        entry_bytes = cfg.fused_entry_bytes(host)
    plan = plan_kv_storage(
        cfg._kv_layer_decls(), page_size=cfg.page_size, fused_draft=placement
    )
    return plan.sub_pool(host).spec, entry_bytes


class TestMambaHostPricing(CustomTestCase):
    """The mamba sub-pool's fused entry is priced from THIS runner's state
    layers, as the pool slices them; the whole model's list would also count
    other pipeline ranks'."""

    def test_a_mamba_region_prices_this_runners_state_layers(self):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        self.enterContext(published_topology())
        cfg.sliding_window_size = None
        cfg.hybrid_gdn_config = None
        cfg.hybrid_kda_config = None
        self.enterContext(patch.object(kcc, "mamba_cache_chunk_size", return_value=64))
        cfg.is_hybrid_swa = False
        cfg.use_mla_backend = False
        cfg.kv_cache_dtype = _DTYPE
        cfg.page_size = 1
        cfg.layer_info = SimpleNamespace(start_layer=2, end_layer=5)
        cfg.model_config = SimpleNamespace(
            get_num_kv_heads=lambda tp, dcp: 2, head_dim=8, v_head_dim=8
        )
        cfg.mambaish_config = SimpleNamespace(
            full_attention_layer_ids=[4],
            mamba2_cache_params=SimpleNamespace(
                layers=[0, 1, 2, 3],
                shape=SimpleNamespace(
                    conv=[(2, 2)], temporal=(2, 2), num_k_heads_per_tp=2
                ),
                dtype=SimpleNamespace(conv=_DTYPE, temporal=_DTYPE),
                is_kda=False,
            ),
        )
        state = DraftStateRegion(
            lane_num=1,
            state=DraftStateGeometry(
                conv_state_shapes=((2, 2),),
                conv_dtype=_DTYPE,
                temporal_state_shape=(2, 2),
                temporal_dtype=_DTYPE,
            ),
        )
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            spec, entry_bytes = _fused_plan(
                cfg,
                FusedDraftPlacement(
                    runners=(RunnerLanes(ranges={"mamba": (0, 1)}),),
                    regions={"mamba": state},
                ),
                "mamba",
            )
        self.assertEqual(spec.layer_num, 2)
        self.assertIs(spec.draft_region, state)
        self.assertEqual(entry_bytes, spec.entry_bytes())


class TestUnifiedSWAHeadGeometry(CustomTestCase):
    """BUG REGRESSION. The hybrid-SWA factories hard-coded a symmetric shape
    on GPU hosts, so an asymmetric model (MiMo-V2-Flash, 192/128) allocated
    192-wide V rows in both sub-pools while the boot solve priced 128."""

    def _configurator(self):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        # The window the runner resolved for its layers.
        cfg.sliding_window_size = 128
        cfg.draft_parallel_layout = None
        cfg.is_hybrid_swa = True
        cfg.is_hybrid_swa_compress = False
        cfg.mambaish_config = None
        cfg.use_mla_backend = False
        cfg.kv_cache_dtype = _DTYPE
        cfg.page_size = 1
        cfg.layer_info = SimpleNamespace(
            start_layer=0,
            end_layer=4,
            full_attention_layer_ids=[0, 3],
            swa_attention_layer_ids=[1, 2],
        )
        cfg.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiMoV2FlashForCausalLM"]),
            sliding_window_size=128,
            get_num_kv_heads=lambda tp, dcp: 4,
            head_dim=192,
            v_head_dim=128,
            get_swa_num_kv_heads=lambda tp: 8,
            swa_head_dim=192,
            swa_v_head_dim=128,
            # The whole model's split; a runner prices its own layer_info.
            swa_attention_layer_ids=[1, 2, 4, 5],
        )
        return cfg

    def test_both_sub_pools_take_the_model_config_geometry(self):
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            g = self._configurator()._unified_swa_head_geometry()
        self.assertEqual((g.head_num, g.head_dim, g.v_head_dim), (4, 192, 128))
        self.assertEqual(
            (g.swa_head_num, g.swa_head_dim, g.swa_v_head_dim), (8, 192, 128)
        )

    def test_the_fused_price_reads_the_same_geometry(self):
        region = DenseDraftRegion(
            lane_num=1, head_num=4, head_dim=64, store_dtype=_DTYPE
        )
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            spec, entry_bytes = _fused_plan(
                self._configurator(),
                FusedDraftPlacement(
                    runners=(RunnerLanes(ranges={"full": (0, 1)}),),
                    regions={"full": region},
                ),
                "full",
            )
        self.assertEqual(
            (spec.layer_num, spec.head_dim, spec.v_head_dim), (2, 192, 128)
        )
        self.assertIs(spec.draft_region, region)
        self.assertEqual(entry_bytes, spec.entry_bytes())

    def test_the_swa_fused_price_reads_this_runners_window_layers(self):
        region = DenseDraftRegion(
            lane_num=1, head_num=4, head_dim=64, store_dtype=_DTYPE
        )
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            spec, entry_bytes = _fused_plan(
                self._configurator(),
                FusedDraftPlacement(
                    runners=(RunnerLanes(ranges={"swa": (0, 1)}),),
                    regions={"swa": region},
                ),
                "swa",
            )
        self.assertEqual(
            (spec.layer_num, spec.head_num, spec.head_dim, spec.v_head_dim),
            (2, 8, 192, 128),
        )
        self.assertIs(spec.draft_region, region)
        self.assertEqual(entry_bytes, spec.entry_bytes())


class TestFusedDraftDecision(CustomTestCase):
    """The target's boot decision over a host whose full sub-pool fuses."""

    def _decide(
        self,
        *,
        algorithm="EAGLE",
        draft_kv_dtype=None,
        host_kv_dtype=_DTYPE,
        kv_cache_dtype_flag="auto",
        draft_backend=None,
        target_backends=("triton", "triton"),
        tp_size=1,
        attn_cp_size=1,
        attn_dp_size=1,
        dcp_size=1,
        hicache=False,
        retraction_backup=None,
        draft_parallel_layout=None,
    ):
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.mem_cache import kv_cache_configurator as kvc
        from sglang.srt.speculative import draft_worker_common

        cfg = kvc.KVCacheConfigurator.__new__(kvc.KVCacheConfigurator)
        # The window the runner resolved for its layers.
        cfg.sliding_window_size = 128
        cfg.draft_parallel_layout = draft_parallel_layout
        cfg.is_hybrid_swa = True
        cfg.is_hybrid_swa_compress = False
        cfg.mambaish_config = None
        cfg.use_mla_backend = False
        cfg.is_draft_worker = False
        cfg.spec_algorithm = SpeculativeAlgorithm.from_string(algorithm)
        # A hybrid sliding-window host: one full and one window layer, whose
        # declarations name the sub-pools the draft may fuse into.
        cfg.layer_info = SimpleNamespace(
            start_layer=0,
            end_layer=2,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
        )
        cfg.model_config = SimpleNamespace(
            is_multi_layer_eagle=False,
            sliding_window_size=128,
            hf_config=SimpleNamespace(architectures=["GptOssForCausalLM"]),
            get_num_kv_heads=lambda tp, dcp: 4,
            head_dim=64,
            v_head_dim=64,
            get_swa_num_kv_heads=lambda tp: 4,
            swa_head_dim=64,
            swa_v_head_dim=64,
        )
        cfg.kv_cache_dtype = host_kv_dtype
        cfg.spec_aux_config = SimpleNamespace(
            eagle_draft_num_layers=1,
            draft_kv_num_layers=1,
            dflash_draft_num_layers=1,
            draft_model_config=SimpleNamespace(
                is_hybrid_swa=False,
                is_deepseek_v4_arch=False,
                num_nextn_predict_layers=None,
                get_num_kv_heads=lambda tp: max(1, 4 // tp),
                head_dim=64,
                v_head_dim=64,
                dtype=torch.bfloat16,
                attention_arch=AttentionArch.MHA,
            ),
        )
        memory = SimpleNamespace(
            enable_unified_memory=True, enable_hierarchical_cache=hicache
        )
        spec = SimpleNamespace(
            speculative_num_steps=1,
            speculative_draft_kv_cache_dtype=draft_kv_dtype,
            speculative_draft_attention_backend=draft_backend,
        )
        with (
            published_topology(
                tp_size=tp_size,
                attn_cp_size=attn_cp_size,
                attn_dp_size=attn_dp_size,
            ),
            patch.object(kvc, "get_memory", return_value=memory),
            patch.object(
                kvc,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_retraction_backup=retraction_backup
                ),
            ),
            patch.object(kvc, "get_spec", return_value=spec),
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
            get_parallel().override(attn_dcp_size=dcp_size),
            override_platform(is_xpu=False, is_hip=False),
        ):
            return cfg._fused_draft_decision()

    def test_a_stateless_draft_is_placed(self):
        self.assertIsNotNone(self._decide().placement)

    def test_fused_placement_uses_resolved_width_for_every_protocol(self):
        from sglang.srt.configs.draft_parallel import DraftParallelLayout

        layout = DraftParallelLayout(tp_group=None, attn_tp_size=2)
        for algorithm in ("EAGLE", "DFLASH", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                decision = self._decide(
                    algorithm=algorithm, tp_size=4, draft_parallel_layout=layout
                )
                self.assertIsNotNone(decision.placement, decision.declined)
                self.assertEqual(decision.placement.region("full").head_num, 2)

    def test_cp_draft_placement_keeps_dp_isolated(self):
        for algorithm, cp, dp, heads in (
            ("DFLASH", 1, 1, 1),
            ("DFLASH", 4, 1, 1),
            ("DFLASH", 2, 2, 4),
            ("EAGLE", 4, 1, 4),
            ("DSPARK", 4, 1, 4),
        ):
            with self.subTest(algorithm=algorithm, cp=cp, dp=dp):
                decision = self._decide(
                    algorithm=algorithm,
                    tp_size=4,
                    attn_cp_size=cp,
                    attn_dp_size=dp,
                )
                self.assertIsNotNone(decision.placement, decision.declined)
                self.assertEqual(decision.placement.region("full").head_num, heads)

    def test_a_draft_kv_dtype_unlike_the_host_fuses_in_its_own_dtype(self):
        """A fused region has typed views of its own, so an explicit draft
        dtype other than the host's fuses, with the region priced and viewed
        in the draft's dtype; one that matches the host's fuses as before."""
        placed = self._decide(draft_kv_dtype="auto")
        self.assertEqual(placed.placement.region("full").resolved_kv_dtype(), _DTYPE)
        for algorithm in ("EAGLE", "DFLASH"):
            with self.subTest(algorithm=algorithm):
                other = self._decide(algorithm=algorithm, draft_kv_dtype="fp8_e4m3")
                self.assertIsNotNone(other.placement, other.declined)
                self.assertEqual(
                    other.placement.region("full").resolved_kv_dtype(),
                    torch.float8_e4m3fn,
                )
                same = self._decide(
                    algorithm=algorithm,
                    draft_kv_dtype="fp8_e4m3",
                    host_kv_dtype=torch.float8_e4m3fn,
                    kv_cache_dtype_flag="fp8_e4m3",
                )
                self.assertEqual(
                    same.placement.region("full").resolved_kv_dtype(),
                    torch.float8_e4m3fn,
                )

    def test_the_region_takes_the_dtype_the_draft_resolves(self):
        """The region's dtype is the one the draft runner resolves for itself
        by the KV dtype rules; the fused decision never asks the backend. A
        DFLASH draft on fa4 resolves its compute dtype over an fp8 host (fa4
        reads K in Q's dtype), on triton the host's fp8; over a bf16 host both
        keep bf16. The same holds when the host's fp8 came from `auto`."""
        for host_kv_dtype, flag in (
            (torch.float8_e4m3fn, "fp8_e4m3"),
            (torch.float8_e4m3fn, "auto"),
            (torch.bfloat16, "auto"),
        ):
            for draft_backend, fp8_region in (
                ("fa4", torch.bfloat16),
                ("triton", None),
            ):
                with self.subTest(host=host_kv_dtype, flag=flag, backend=draft_backend):
                    decision = self._decide(
                        algorithm="DFLASH",
                        draft_backend=draft_backend,
                        host_kv_dtype=host_kv_dtype,
                        kv_cache_dtype_flag=flag,
                    )
                    self.assertIsNotNone(decision.placement, decision.declined)
                    expected = (
                        host_kv_dtype
                        if host_kv_dtype == torch.bfloat16 or fp8_region is None
                        else fp8_region
                    )
                    self.assertEqual(
                        decision.placement.region("full").resolved_kv_dtype(),
                        expected,
                    )

    def test_an_auto_draft_dtype_sizes_the_region_as_the_loaded_draft_resolves(self):
        """BUG REGRESSION. An explicit draft KV dtype of `auto` sized the region
        without the draft's quant config, in its compute dtype, while the
        loaded draft, whose quant config declares an FP8 KV cache, resolved
        FP8 and failed to bind. Until the draft loads, its `auto` takes the
        host's dtype, as no flag does; an explicit dtype is its own."""
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype

        loaded_draft = SimpleNamespace(
            quant_config=Fp8Config(
                is_checkpoint_fp8_serialized=True, kv_cache_quant_algo="FP8"
            )
        )
        for draft_kv_dtype in (None, "auto", "fp8_e4m3"):
            with self.subTest(draft_kv_dtype=draft_kv_dtype):
                decision = self._decide(
                    draft_kv_dtype=draft_kv_dtype,
                    host_kv_dtype=torch.float8_e4m3fn,
                    kv_cache_dtype_flag="fp8_e4m3",
                )
                self.assertIsNotNone(decision.placement, decision.declined)
                # What the draft worker resolves once its model has loaded.
                _, resolved = configure_kv_cache_dtype(
                    server_args_kv_cache_dtype="fp8_e4m3",
                    model=loaded_draft,
                    model_dtype=_DTYPE,
                    is_draft_worker=True,
                    is_dflash=False,
                    speculative_draft_attention_backend=None,
                    speculative_draft_kv_cache_dtype=draft_kv_dtype,
                )
                self.assertEqual(resolved, torch.float8_e4m3fn)
                self.assertEqual(
                    decision.placement.region("full").resolved_kv_dtype(), resolved
                )

    def test_no_draft_backend_keeps_the_private_pool(self):
        """A fused draft reads its rows through the target's views and id
        spaces, so the draft's backend, named or inherited, never decides
        whether it fuses."""
        self.assertIsNotNone(
            self._decide(algorithm="EAGLE", draft_backend="flashmla").placement
        )
        for target_backends in (("fa4", "fa4"), ("fa3", "flashmla")):
            for algorithm in ("EAGLE", "DSPARK"):
                decision = self._decide(
                    algorithm=algorithm, target_backends=target_backends
                )
                self.assertIsNotNone(decision.placement, (algorithm, target_backends))

    def test_a_fa4_dspark_draft_fuses(self):
        decision = self._decide(algorithm="DSPARK", draft_backend="fa4")
        self.assertIsNotNone(decision.placement)

    def test_a_trtllm_mha_draft_fuses(self):
        """trtllm_mha refills its graph page table to what each replay reads
        and widens its eager table by the draft block, so its draft fuses."""
        # Kimi-Linear + DSPARK on SM100, where a model hook declares trtllm_mha.
        for algorithm in ("EAGLE", "DFLASH", "DSPARK"):
            decision = self._decide(algorithm=algorithm, draft_backend="trtllm_mha")
            self.assertIsNotNone(decision.placement, algorithm)
        # A draft with no backend of its own inherits the target's trtllm_mha.
        decision = self._decide(target_backends=("trtllm_mha", "trtllm_mha"))
        self.assertIsNotNone(decision.placement)

    def test_dcp_keeps_the_private_pool(self):
        """Under --dcp-size > 1 each rank's host rows hold only its share of
        the widened id space, while the draft, outside the DCP group, reads
        every token."""
        for algorithm in ("DSPARK", "EAGLE"):
            declined = self._decide(algorithm=algorithm, dcp_size=2)
            self.assertIsNone(declined.placement, algorithm)
            self.assertIn("--dcp-size 2", declined.declined)

    def test_hierarchical_cache_keeps_the_private_pool(self):
        """HiCache builds the draft's host pool off its own device pool, which
        a fused draft does not have; DSPARK + HiCache keeps working on the
        private pool."""
        declined = self._decide(algorithm="DSPARK", hicache=True)
        self.assertIsNone(declined.placement)
        self.assertIn("--enable-hierarchical-cache", declined.declined)

    def test_host_pool_retraction_keeps_the_private_pool(self):
        """Host-pool decode retraction builds the draft's host pool off its
        own device pool too; DSPARK with it keeps the private pool."""
        declined = self._decide(algorithm="DSPARK", retraction_backup="host_pool")
        self.assertIsNone(declined.placement)
        self.assertIn(
            "--disaggregation-decode-retraction-backup=host_pool", declined.declined
        )


if __name__ == "__main__":
    unittest.main()
