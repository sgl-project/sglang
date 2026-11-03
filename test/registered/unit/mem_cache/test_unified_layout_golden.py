"""Golden byte layout of the unified KV pool, for every sub-pool chain.

PD prefill and decode peers address each other's page envelopes by physical
page id, and HiCache copies whole envelopes and state entries, so the values
pinned here are a wire format: chain order, entry bytes, part offsets, view
strides, the slot-0 sink, slot bounds and the empty-pool capacities. A changed
literal is a change of the PD/HiCache byte layout; the PR that changes one must
say so and why. Nothing is recomputed from the implementation at test time:
the literals were produced once and are compared as data.
"""

import json
import re
import unittest
from pathlib import Path

import msgspec
import torch

from sglang.srt.mem_cache.kv.layout.base import check_physical_id_bound
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
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
        counts={h: [1] for h in hosts}, regions=regions
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
    # Entries align32(2 x (16 B K + 16 B V)) = 64 B; total 64 x 64 + 64 x 64 = 8192;
    # the sink is one 64 B entry, so min_slot_index 1 and max_slots 8192 // 64 = 128.
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
        fused_draft=_draft("full", "swa"),
        full_tokens=64 * 16,
        swa_tokens=64 * 16,
    ),
    "swa_p16_bytes": _swa(page_size=16, byte_budget=1 << 20),
    # The state entry stays unaligned: 2 x (3 x 8 bf16 + 1 x 4 x 8 fp32) = 352 B;
    # total 64 x 64 + 8 x 352 = 6912; the 352 B sink puts full's first slot at 6.
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
    # Total 64 x 64 + 64 x 64 + 8 x 352 = 11008; window and full max_slots 172.
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
        fused_draft=_draft("swa"),
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
# floor binds only when the state slots alone do not cover it, hence two slots.
_FLOOR_CONFIGS = {
    "swa_p16": _CONFIGS["swa_p16"],
    "mamba_p16": _mamba(page_size=16, state_slots=2),
    "tri_p16": _CONFIGS["tri_p16"],
}

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

    def test_bs1_floor_refuses_one_byte_below_and_builds_at_it(self):
        """The retract loop's last-request guarantee: a buffer one byte short of
        the itemized floor fails at boot, naming every term; at the floor it builds."""
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
                build_unified_bundle(
                    msgspec.structs.replace(geometry, byte_budget=budget)
                )

    def test_physical_ids_stay_within_int32(self):
        check_physical_id_bound(sub_pool="full", n_rows=2**31 - 1)
        with self.assertRaises(AssertionError):
            check_physical_id_bound(sub_pool="full", n_rows=2**31)


if __name__ == "__main__":
    unittest.main()
