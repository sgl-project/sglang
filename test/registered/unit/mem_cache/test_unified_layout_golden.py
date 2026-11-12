"""Golden byte layout of the unified KV pool, for every sub-pool chain.

PD prefill and decode peers address each other's page envelopes by physical
page id, and HiCache copies whole envelopes and state entries, so the values
pinned here include the wire format (chain order, entry bytes, part offsets,
view strides and the slot-0 sink) and capacity results (slot bounds and empty
pool capacity). Total-budget sizing charges the sink once and fixed request
storage outside the arena; capacity literals include that accounting. Nothing is recomputed from the implementation at test time:
the literals were produced once and are compared as data.
"""

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.kv.decl import (
    FP8_DS_MLA,
    FullLayerParts,
    IndexKeyGeometry,
    MHAGeometry,
    MLAGeometry,
    MLARowFormat,
    StateGeometry,
    kv_layer_decls,
    with_state_ple,
)
from sglang.srt.mem_cache.kv.layout.base import check_physical_id_bound
from sglang.srt.mem_cache.kv.plan import sub_pool_names
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_FULL,
    LAYER_STATE,
    LAYER_WINDOW,
    DenseDraftRegion,
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    DraftMLAGeometry,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
    PlacementContext,
    place_fused_draft,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    bind_test_draft,
    build_declared_bundle,
    build_unified_bundle,
    fused_draft_layout_summary,
    setup_allocator_context,
    unified_layout_summary,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_DENSE_DRAFT = DenseDraftRegion(
    lane_num=1, head_num=1, head_dim=8, store_dtype=torch.float16
)
_STATE_DRAFT = DraftStateRegion(
    lane_num=1,
    state=DraftStateGeometry(
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(1, 4, 8),
        temporal_dtype=torch.float32,
    ),
)


def _draft(*hosts: str) -> FusedDraftPlacement:
    regions = {h: _STATE_DRAFT if h == "mamba" else _DENSE_DRAFT for h in hosts}
    return FusedDraftPlacement.from_counts(
        counts={h: [1] for h in hosts},
        regions=regions,
        # The target's window sub-pool hosts as a window.
        hosted={h: "window" for h in hosts if h.startswith("window_")},
    )


def _swa(**kw) -> UnifiedGeometry:
    return UnifiedGeometry(
        chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3), **kw
    )


def _mamba(**kw) -> UnifiedGeometry:
    return UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3), **kw
    )


def _tri(**kw) -> UnifiedGeometry:
    return UnifiedGeometry(
        chain="tri",
        full_layer_ids=kw.pop("full_layer_ids", (0, 1)),
        swa_layer_ids=kw.pop("swa_layer_ids", (2, 3)),
        state_layer_ids=kw.pop("state_layer_ids", (0, 2)),
        **kw,
    )


_CONFIGS = {
    # Entries align32(2 x (16 B K + 16 B V)) = 64 B. The arena contains
    # 64 full + 64 window entries and one shared 64 B sink.
    "swa_p1": _swa(page_size=1),
    "swa_p16": _swa(page_size=16, full_tokens=64 * 16, swa_tokens=64 * 16),
    "swa_p64": _swa(page_size=64, full_tokens=64 * 64, swa_tokens=64 * 64),
    # Asymmetric rows; the full entry's raw size (3 x 48 B) is not a multiple of 32.
    "swa_p16_asym": UnifiedGeometry(
        chain="swa",
        full_layer_ids=(0, 1, 2),
        swa_layer_ids=(3, 4),
        page_size=16,
        v_head_dim=16,
        swa_head_num=2,
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "swa_p16_fp8": _swa(
        page_size=16,
        head_num=2,
        swa_head_num=2,
        kv_cache_dtype=torch.float8_e4m3fn,
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "swa_p16_draft_full": _swa(
        page_size=16,
        fused_draft=_draft("full"),
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "swa_p16_draft_full_swa": _swa(
        page_size=16,
        fused_draft=_draft("full", "window_32"),
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "swa_p16_bytes": _swa(page_size=16, byte_budget=1 << 20),
    # The 96 B conv block is padded to 128 B before the two 128 B temporal rows.
    # The 384 B state entry/sink preserves 64 B temporal alignment across slots.
    "mamba_p1": _mamba(page_size=1),
    "mamba_p16": _mamba(page_size=16, full_tokens=64 * 16),
    "mamba_p64": _mamba(page_size=64, full_tokens=64 * 64),
    "mamba_p16_draft_full": _mamba(
        page_size=16, fused_draft=_draft("full"), full_tokens=64 * 16
    ),
    "mamba_p16_draft_state": _mamba(
        page_size=16, fused_draft=_draft("mamba"), full_tokens=64 * 16
    ),
    "mamba_p16_bytes": _mamba(page_size=16, byte_budget=1 << 20),
    "mamba_p16_slice_axis": _mamba(
        page_size=16, conv_slice_axis=1, full_tokens=64 * 16
    ),
    "mla_p1": _mamba(page_size=1, mla=True),
    "mla_p16": _mamba(page_size=16, mla=True, full_tokens=64 * 16),
    "mla_p64": _mamba(page_size=64, mla=True, full_tokens=64 * 64),
    "mla_p16_draft_full": _mamba(
        page_size=16, mla=True, fused_draft=_draft("full"), full_tokens=64 * 16
    ),
    # Full, window and state share one sink, sized to their largest unit.
    "tri_p1": _tri(page_size=1),
    "tri_p16": _tri(page_size=16, full_tokens=64 * 16, swa_tokens=64 * 16),
    "tri_p64": _tri(page_size=64, full_tokens=64 * 64, swa_tokens=64 * 64),
    "tri_p16_asym": _tri(
        page_size=16,
        v_head_dim=16,
        swa_head_num=2,
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "tri_p16_draft_full": _tri(
        page_size=16,
        fused_draft=_draft("full"),
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "tri_p16_draft_swa": _tri(
        page_size=16,
        fused_draft=_draft("window_32"),
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "tri_p16_draft_state": _tri(
        page_size=16,
        fused_draft=_draft("mamba"),
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "tri_p16_slice_axis": _tri(
        page_size=16,
        conv_slice_axis=1,
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    # A pipeline stage holding layers 2 and 3 of a four-layer model.
    "tri_pp_stage": _tri(
        page_size=16,
        full_layer_ids=(2,),
        swa_layer_ids=(3,),
        state_layer_ids=(2,),
        start_layer=2,
        end_layer=4,
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
}

# One configuration per chain whose bs=1 floor the golden pins. The full + state
# floor uses the three slots required by the radix state strategy.
_FLOOR_CONFIGS = {
    "swa_p16": _CONFIGS["swa_p16"],
    "mamba_p16": _mamba(page_size=16, state_slots=3),
    "tri_p16": _CONFIGS["tri_p16"],
}


# -- fused draft placements ------------------------------------------------------
#
# Every fused-draft placement the unified pool produces, through the placement
# production runs (`place_fused_draft`) over a target's declarations, then the
# production binders per draft runner. The summary is read off the built views,
# so it pins bytes, not the classes that lay them out.

_BF16, _FP16, _FP8, _U8 = (
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.uint8,
)
_F32 = torch.float32


def _mha(heads=1, dim=8, v=None, dtype=_BF16):
    return MHAGeometry(
        head_num=heads,
        head_dim=dim,
        v_head_dim=dim if v is None else v,
        kv_cache_dtype=dtype,
    )


def _mla(dtype=_BF16, rank=16, rope=8):
    return MLAGeometry(kv_lora_rank=rank, qk_rope_head_dim=rope, kv_cache_dtype=dtype)


def _state(conv=((3, 8),), conv_dtype=_BF16, temporal=(1, 4, 8), temporal_dtype=_F32):
    return StateGeometry(
        conv_state_shapes=conv,
        conv_dtype=conv_dtype,
        temporal_state_shape=temporal,
        temporal_dtype=temporal_dtype,
    )


def _state_params(layer_ids):
    return SimpleNamespace(layers=list(layer_ids))


class _Target(msgspec.Struct, frozen=True, kw_only=True):
    """A target model's declarations and the counts its pool is sized by."""

    full: Tuple[int, ...]
    full_geometry: object
    window: Tuple[int, ...] = ()
    window_geometry: object = None
    window_len: Optional[int] = None
    state: Tuple[int, ...] = ()
    state_geometry: object = None
    full_parts: Tuple[object, ...] = ()
    page_size: int = 16

    def decls(self):
        return kv_layer_decls(
            full_layer_ids=self.full,
            full_geometry=self.full_geometry,
            window_layer_ids=self.window,
            window_geometry=self.window_geometry,
            window=self.window_len,
            state_layer_ids=self.state,
            state_geometry=self.state_geometry,
            checkpoint_interval=64 if self.state else None,
            full_layer_parts=self.full_parts,
        )


_SWA = _Target(
    full=(0, 1),
    full_geometry=_mha(2, 8),
    window=(2, 3),
    window_geometry=_mha(1, 8),
    window_len=32,
)
_SWA_FP8 = msgspec.structs.replace(
    _SWA, full_geometry=_mha(2, 8, dtype=_FP8), window_geometry=_mha(2, 8, dtype=_FP8)
)
# An MLA trunk with KDA layers: three conv streams (q, k, v) per state layer.
_MLA_KDA = _Target(
    full=(1, 3),
    full_geometry=_mla(),
    state=(0, 2),
    state_geometry=_state(conv=((3, 16), (3, 16), (3, 16)), temporal=(2, 8, 8)),
)
# One 48 B latent row per token: the host block ends off the 32 B grid.
_MLA_ONE = _Target(full=(1,), full_geometry=_mla(), state=(0,), state_geometry=_state())
# A GDN trunk whose state also keeps a short-conv part.
_GDN = _Target(
    full=(3,),
    full_geometry=_mha(2, 16),
    state=(0, 1, 2),
    state_geometry=with_state_ple(
        _state(conv=((3, 24),), temporal=(2, 8, 8)),
        short_conv_layer_ids=(0, 2),
        short_conv_state_shape=(2, 16),
    ),
)
_TRI = _Target(
    full=(0,),
    full_geometry=_mha(1, 8),
    window=(1,),
    window_geometry=_mha(1, 8),
    window_len=32,
    state=(2,),
    state_geometry=_state(),
)
# A sparse-attention trunk: layer-major FP8 latent rows with block scales and
# the indexer keys beside them, in one composite unit, beside KDA state.
_DSA = _Target(
    full=(1, 3),
    full_geometry=MLAGeometry(
        kv_lora_rank=128,
        qk_rope_head_dim=8,
        kv_cache_dtype=_FP8,
        row_format=MLARowFormat(name=FP8_DS_MLA, row_bytes=160, scale_ue8m0=True),
        layer_major=True,
    ),
    full_parts=(
        FullLayerParts(layer_ids=(1, 3), geometry=IndexKeyGeometry(index_head_dim=128)),
    ),
    state=(0, 2),
    state_geometry=_state(conv=((3, 16), (3, 16), (3, 16)), temporal=(2, 8, 8)),
)


class _Draft(msgspec.Struct, frozen=True, kw_only=True):
    """A draft checkpoint as the placement sees it, the runners serving it, the
    widths and dtype it runs at, and its runners' model layer ids."""

    num_layers: int
    full: Tuple[int, ...] = ()
    full_geometry: object = None
    window: Tuple[int, ...] = ()
    window_geometry: object = None
    window_len: Optional[int] = None
    state: Tuple[int, ...] = ()
    state_geometry: object = None
    num_depths: int = 1
    num_runners: int = 1
    dcp_size: int = 1
    kv_dtype: torch.dtype = _BF16

    def profile(self):
        kinds = {}
        if self.full:
            kinds[LAYER_FULL] = DraftLayerSet(
                layer_ids=self.full, geometry=self.full_geometry
            )
        if self.window:
            kinds[LAYER_WINDOW] = DraftLayerSet(
                layer_ids=self.window,
                geometry=self.window_geometry,
                window=self.window_len,
            )
        if self.state:
            kinds[LAYER_STATE] = DraftLayerSet(
                layer_ids=self.state, geometry=self.state_geometry
            )
        return DraftKVProfile(
            num_layers=self.num_layers, num_depths=self.num_depths, kinds=kinds
        )

    def runner_layers(self, runner: int) -> Tuple[int, ...]:
        """A runner's attention layer ids: every layer of a replicated head,
        its own depth of a per-depth head (when the depth has attention)."""
        if self.num_depths <= 1:
            return tuple(sorted(self.full + self.window))
        return tuple(i for i in (runner,) if i in self.full + self.window)


def _kv(heads, dim, v=None):
    return DraftKVGeometry(
        head_num=heads, head_dim=dim, v_head_dim=dim if v is None else v
    )


_LATENT = DraftMLAGeometry(kv_lora_rank=16, qk_rope_head_dim=8)
_DRAFT_STATE = DraftStateGeometry(
    conv_state_shapes=((3, 16), (3, 16), (3, 16)),
    conv_dtype=_BF16,
    temporal_state_shape=(2, 8, 8),
    temporal_dtype=_F32,
)

# name -> (target, draft or None)
_FUSED = {
    # No draft: the targets the shared layout engine will lay out.
    "plain_swa": (_SWA, None),
    "plain_mla_kda": (_MLA_KDA, None),
    "plain_gdn": (_GDN, None),
    "plain_dsa": (_DSA, None),
    # MHA drafts: an asymmetric V, then rows whose 3 x 8 x 2 = 48 B K row and
    # 3 x 4 x 2 = 24 B V row pad to the 16 B row grid.
    "mha_asym_v_in_mha": (
        _SWA,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 16, 8)),
    ),
    "mha_padded_in_mha": (
        _SWA,
        _Draft(num_layers=2, full=(0, 1), full_geometry=_kv(3, 8, 4)),
    ),
    "mha_in_mla": (_MLA_KDA, _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 8))),
    "mla_in_mla": (_MLA_KDA, _Draft(num_layers=1, full=(0,), full_geometry=_LATENT)),
    "mha_after_odd_host": (
        _MLA_ONE,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(1, 8)),
    ),
    "replicas2_odd_host": (
        _MLA_ONE,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(1, 8), dcp_size=2),
    ),
    "mla_in_mha": (_SWA, _Draft(num_layers=1, full=(0,), full_geometry=_LATENT)),
    "mha_in_gdn": (_GDN, _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 16))),
    # Per-depth heads with recurrent state, in a KDA trunk's state entries.
    "state_per_depth": (
        _MLA_KDA,
        _Draft(
            num_layers=1,
            full=(0, 1),
            full_geometry=_kv(1, 8),
            state=(0, 1),
            state_geometry=_DRAFT_STATE,
            num_depths=2,
            num_runners=2,
        ),
    ),
    "state_mixed_dtypes": (
        _GDN,
        _Draft(
            num_layers=1,
            full=(0, 1),
            full_geometry=_kv(2, 16),
            state=(0, 1),
            state_geometry=DraftStateGeometry(
                conv_state_shapes=((3, 24), (2, 8)),
                conv_dtype=_FP16,
                temporal_state_shape=(2, 8, 8),
                temporal_dtype=_F32,
            ),
            num_depths=2,
            num_runners=2,
        ),
    ),
    # Draft dtypes other than the host's.
    "fp8_draft_in_bf16": (
        _SWA,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 8), kv_dtype=_FP8),
    ),
    "bf16_draft_in_fp8": (
        _SWA_FP8,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 8)),
    ),
    "fp8_latent_in_bf16": (
        _MLA_KDA,
        _Draft(num_layers=1, full=(0,), full_geometry=_LATENT, kv_dtype=_FP8),
    ),
    # Rows replicated over every logical token of an owner DCP page.
    "replicas2_mha_host": (
        _GDN,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 8), dcp_size=2),
    ),
    "replicas4_mha_host": (
        _GDN,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 8), dcp_size=4),
    ),
    "replicas2_mla_host": (
        _MLA_KDA,
        _Draft(num_layers=1, full=(0,), full_geometry=_LATENT, dcp_size=2),
    ),
    "replicas4_mla_host": (
        _MLA_KDA,
        _Draft(num_layers=1, full=(0,), full_geometry=_kv(2, 8), dcp_size=4),
    ),
    # Lanes across runners: a replicated head on three runners, a per-depth
    # head on two, and a per-depth head mixing full and window depths.
    "replicated_three_runners": (
        _GDN,
        _Draft(num_layers=2, full=(0, 1), full_geometry=_kv(1, 8), num_runners=3),
    ),
    "per_depth_two_runners": (
        _MLA_KDA,
        _Draft(
            num_layers=1,
            full=(0, 1),
            full_geometry=_LATENT,
            num_depths=2,
            num_runners=2,
        ),
    ),
    "per_depth_full_and_window": (
        _SWA,
        _Draft(
            num_layers=1,
            full=(0, 2),
            full_geometry=_kv(2, 8),
            window=(1,),
            window_geometry=_kv(1, 8),
            window_len=32,
            num_depths=3,
            num_runners=3,
        ),
    ),
    # Window layers: one sharing the target's window, ones in bands of their own.
    "matched_window": (
        _SWA,
        _Draft(
            num_layers=2,
            full=(0,),
            full_geometry=_kv(2, 8),
            window=(1,),
            window_geometry=_kv(1, 8),
            window_len=32,
        ),
    ),
    "unmatched_window_mha": (
        _TRI,
        _Draft(
            num_layers=2,
            full=(0,),
            full_geometry=_kv(1, 8),
            window=(1,),
            window_geometry=_kv(2, 8),
            window_len=64,
        ),
    ),
    "unmatched_window_mla": (
        _MLA_KDA,
        _Draft(
            num_layers=2,
            full=(0,),
            full_geometry=_LATENT,
            window=(1,),
            window_geometry=_LATENT,
            window_len=128,
        ),
    ),
}


def _place(target: _Target, draft: _Draft) -> FusedDraftPlacement:
    decls = target.decls()
    windows = tuple(
        sorted(
            {target.window_len} if target.window else set(),
            key=lambda w: w,
        )
    )
    decision = place_fused_draft(
        profile=draft.profile(),
        num_runners=draft.num_runners,
        ctx=PlacementContext(
            host_names=sub_pool_names(decls),
            target_windows=windows,
            dcp_size=draft.dcp_size,
        ),
        store_dtype=_U8 if draft.kv_dtype == _FP8 else draft.kv_dtype,
        kv_dtype=draft.kv_dtype,
    )
    assert decision.placement is not None, decision.declined
    return decision.placement


def build_fused_case(name: str):
    """The built target pool of a case, its placement and its runners' bindings."""
    target, draft = _FUSED[name]
    placement = None if draft is None else _place(target, draft)
    bundle = build_declared_bundle(
        target.decls(),
        page_size=target.page_size,
        full_tokens=8 * target.page_size,
        window_tokens=8 * target.page_size if target.window else None,
        state_slots=4 if target.state else 0,
        state_cache_params=_state_params(target.state) if target.state else None,
        fused_draft=placement,
    )
    bindings = []
    if draft is not None:
        for runner in range(draft.num_runners):
            bindings.append(
                bind_test_draft(
                    bundle,
                    placement,
                    runner,
                    kv_layer_ids=draft.runner_layers(runner),
                    window_layer_ids=draft.window,
                    state_layer_ids=draft.state,
                    num_depths=draft.num_depths,
                )
            )
    return bundle, placement, bindings


def fused_case_summary(name: str) -> dict:
    bundle, placement, bindings = build_fused_case(name)
    return fused_draft_layout_summary(bundle, placement, bindings)


_GOLDEN_PATH = Path(__file__).with_name("unified_layout_golden.json")


class TestUnifiedLayoutGolden(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(_GOLDEN_PATH.read_text())

    def setUp(self):
        setup_allocator_context()

    def test_layout_of_every_chain_matches_the_wire_format(self):
        layouts = self.golden["layouts"]
        self.assertEqual(set(layouts), set(_CONFIGS))
        for name, geometry in _CONFIGS.items():
            with self.subTest(config=name):
                summary = unified_layout_summary(build_unified_bundle(geometry))
                self.assertEqual(json.loads(json.dumps(summary)), layouts[name])

    def test_retention_floor_and_minimum_usable_budget(self):
        """Retention must fit, and the boot targets must leave a usable FULL page.
        A retention floor alone can leave zero FULL pages after reserving STATE.
        """
        self.assertEqual(set(self.golden["bs1_floors"]), set(_FLOOR_CONFIGS))
        for name, floor in self.golden["bs1_floors"].items():
            geometry = _FLOOR_CONFIGS[name]
            budget = floor["budget_at_floor"]
            with self.subTest(config=name):
                with self.assertRaises(RuntimeError) as ctx:
                    build_unified_bundle(
                        msgspec.structs.replace(geometry, byte_budget=budget - 1)
                    )
                for term, nbytes in floor["terms"]:
                    self.assertRegex(str(ctx.exception), re.escape(f"{term}={nbytes}"))
                minimum = floor["minimum_usable_budget"]
                if minimum > budget:
                    with self.assertRaisesRegex(
                        RuntimeError, "cannot serve one request"
                    ):
                        build_unified_bundle(
                            msgspec.structs.replace(geometry, byte_budget=minimum - 1)
                        )
                build_unified_bundle(
                    msgspec.structs.replace(geometry, byte_budget=minimum)
                )

    def test_every_fused_draft_layout_matches(self):
        """Every byte a target and its fused draft runners address: bands,
        chain, sink, placement, routed views, state blocks and id spaces."""
        layouts = self.golden["fused_draft_layouts"]
        self.assertEqual(set(layouts), set(_FUSED))
        for name in _FUSED:
            with self.subTest(case=name):
                summary = fused_case_summary(name)
                self.assertEqual(json.loads(json.dumps(summary)), layouts[name])

    def test_draft_bands_are_windows_only(self):
        """Full layers always ride in the target's full sub-pool and state in
        its state sub-pool, so a draft's own band only ever carries window
        layers whose length no target window matches."""
        for name, (target, draft) in _FUSED.items():
            if draft is None:
                continue
            with self.subTest(case=name):
                placement = _place(target, draft)
                for band_name, band in placement.new_bands.items():
                    self.assertEqual(band.kind, LAYER_WINDOW, band_name)
                    self.assertTrue(band_name.startswith("window_"), band_name)
                    self.assertNotIn(band.window, (target.window_len,))

    def test_physical_ids_stay_within_int32(self):
        check_physical_id_bound(sub_pool="full", n_rows=2**31 - 1)
        with self.assertRaises(AssertionError):
            check_physical_id_bound(sub_pool="full", n_rows=2**31)


if __name__ == "__main__":
    unittest.main()
