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
    so do asymmetric K/V rows unless every resolved backend, the draft's
    included, carries v_head_dim through to the kernel;
  - so does a draft whose attention backend is off the translated MHA rails
    (it would read the fused rows with virtual ids), a draft under
    --dcp-size > 1 (each rank's rows hold only its share of the tokens), an
    explicit draft KV dtype unlike the host's, and a draft under HiCache or
    host-pool decode retraction (they build the draft's host pool off a
    device pool of its own);
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

from sglang.srt.arg_groups.kv_cache_hook import ASYMMETRIC_KV_BACKENDS
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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


def _place(
    profile,
    num_runners=1,
    asymmetric_rows_ok=False,
    host_names=_HOSTS,
    target_window=None,
    draft_backends=("triton",),
):
    return place_fused_draft(
        profile=profile,
        num_runners=num_runners,
        ctx=PlacementContext(
            host_names=host_names,
            target_window=target_window,
            asymmetric_rows_ok=asymmetric_rows_ok,
            draft_backends=draft_backends,
        ),
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

    def test_a_window_draft_needs_a_rail_carrying_draft_backend(self):
        """Only the Triton multi-step draft backend builds the per-step window
        rails; any other draft backend would sink-write the window layers."""
        decision = _place(
            _profile(window_layer_ids=(0,), window=128),
            target_window=128,
            draft_backends=("fa3",),
        )
        self.assertIsNone(decision.placement)
        self.assertIn("['fa3']", decision.declined)
        self.assertIn("triton", decision.declined)
        # A full-only draft never consults the rail rule.
        self.assertIsNotNone(_place(_profile(), draft_backends=("fa3",)).placement)

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

    def test_asymmetric_and_misaligned_rows_decline(self):
        for profile, reason in (
            (_profile(head_dim=64, v_head_dim=32), "asymmetric"),
            # 4 heads x 5 dims x 2 B = 40 B: not a 16-B-aligned entry part.
            (_profile(head_dim=5, v_head_dim=5), "40 B"),
        ):
            decision = _place(profile)
            self.assertIsNone(decision.placement)
            self.assertIn(reason, decision.declined)

    def test_asymmetric_rows_fuse_with_their_v_width_when_backends_allow(self):
        placement = _place(
            _profile(head_dim=64, v_head_dim=32), asymmetric_rows_ok=True
        ).placement
        self.assertIsNotNone(placement)
        self.assertEqual(placement.region("full").resolved_v_head_dim(), 32)
        self.assertEqual(placement.region("full").entry_bytes(), 4 * (64 + 32) * 2)
        # Symmetric rows never consult the rule.
        self.assertIsNotNone(_place(_profile(), asymmetric_rows_ok=False).placement)

    def test_a_misaligned_v_row_declines_even_when_backends_allow(self):
        # K rows are 4 x 64 x 2 B = 512 B; V rows 4 x 5 x 2 B = 40 B are not a
        # 16-B-aligned entry part.
        decision = _place(_profile(head_dim=64, v_head_dim=5), asymmetric_rows_ok=True)
        self.assertIsNone(decision.placement)
        self.assertIn("40 B", decision.declined)

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


class TestAsymmetricBackendRule(CustomTestCase):
    """The rule reads the TARGET's resolved backends and the draft's explicit
    one (unset inherits the target's); one shared-head_dim backend anywhere
    disqualifies asymmetric rows."""

    def _carries(self, *, backends, draft_backend=None):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        with (
            patch.object(kcc, "attention_backends", return_value=tuple(backends)),
            patch.object(
                kcc,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_attention_backend=draft_backend
                ),
            ),
        ):
            return cfg._draft_backends_carry_v_head_dim()

    def test_every_split_stride_backend_qualifies(self):
        for backend in sorted(ASYMMETRIC_KV_BACKENDS):
            self.assertTrue(self._carries(backends=(backend,)), backend)

    def test_a_shared_head_dim_backend_anywhere_disqualifies(self):
        self.assertNotIn("flashinfer", ASYMMETRIC_KV_BACKENDS)
        self.assertFalse(self._carries(backends=("flashinfer",)))
        self.assertFalse(
            self._carries(backends=("triton",), draft_backend="flashinfer")
        )


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


class TestMambaHostPricing(CustomTestCase):
    """The mamba sub-pool's fused entry is priced from THIS runner's state
    layers, as the mamba factory slices them; the whole model's list would
    also count other pipeline ranks'."""

    def test_a_mamba_region_prices_this_runners_state_layers(self):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.layer_info = SimpleNamespace(start_layer=2, end_layer=4)
        cfg.mambaish_config = SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                layers=[0, 1, 2, 3],
                shape=SimpleNamespace(conv=[(2, 2)], temporal=(2, 2)),
                dtype=SimpleNamespace(conv=_DTYPE, temporal=_DTYPE),
            )
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
        spec = cfg._mamba_host_spec(state)
        self.assertEqual(spec.layer_num, 2)
        self.assertIs(spec.draft_region, state)


class TestUnifiedSWAHeadGeometry(CustomTestCase):
    """BUG REGRESSION. The hybrid-SWA factories hard-coded a symmetric shape
    on GPU hosts, so an asymmetric model (MiMo-V2-Flash, 192/128) allocated
    192-wide V rows in both sub-pools while the boot solve priced 128."""

    def _configurator(self):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_hybrid_swa = True
        cfg.is_hybrid_swa_compress = False
        cfg.use_mla_backend = False
        cfg.kv_cache_dtype = _DTYPE
        cfg.layer_info = SimpleNamespace(
            full_attention_layer_ids=[0, 3], swa_attention_layer_ids=[1, 2]
        )
        cfg.model_config = SimpleNamespace(
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
            spec = self._configurator()._full_host_spec(region)
        self.assertEqual(
            (spec.layer_num, spec.head_dim, spec.v_head_dim), (2, 192, 128)
        )

    def test_the_swa_fused_price_reads_this_runners_window_layers(self):
        region = DenseDraftRegion(
            lane_num=1, head_num=4, head_dim=64, store_dtype=_DTYPE
        )
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            spec = self._configurator()._swa_host_spec(region)
        self.assertEqual(
            (spec.layer_num, spec.head_num, spec.head_dim, spec.v_head_dim),
            (2, 8, 192, 128),
        )


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
        dcp_size=1,
        hicache=False,
        retraction_backup=None,
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
        cfg.model_config = SimpleNamespace(
            is_multi_layer_eagle=False, sliding_window_size=None
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
                get_num_kv_heads=lambda tp: 4,
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
            get_parallel().override(attn_tp_size=1, attn_dcp_size=dcp_size),
            override_platform(is_xpu=False, is_hip=False),
        ):
            return cfg._fused_draft_decision()

    def test_a_stateless_draft_is_placed(self):
        self.assertIsNotNone(self._decide().placement)

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
        only triton, flashinfer, fa3 and trtllm_mha carry on every draft path.
        The draft's backend is the published one, a model hook's declaration
        included; otherwise it is what the draft runner inherits from the
        target."""
        declined = self._decide(algorithm="DSPARK", draft_backend="fa4")
        self.assertIsNone(declined.placement)
        self.assertIn("fa4", declined.declined)
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
