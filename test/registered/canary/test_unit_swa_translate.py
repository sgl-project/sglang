"""Host-side unit tests for SWA full -> swa slot-index translation.

The SWA canary group on a real ``SWAKVPool`` lives on the swa sub-pool's
slot index space, but ``forward_batch.out_cache_loc`` and
``req_to_token_pool.req_to_token`` deliver full-pool indices. The
plan-side translation through ``full_to_swa_index_mapping`` (captured
into ``CanaryBufferGroup.swa_index_lut``) must:

- remap the four plan slot-index fields,
- preserve ``-1`` (chain head) and ``-2`` (skip-chain) sentinels,
- propagate ``-1`` for positions whose LUT entry is the
  not-in-window sentinel,
- be a strict no-op when the group has no LUT (FULL group / MHA / MLA
  / DSV4 fallback).
"""

from __future__ import annotations

import unittest

import torch

from sglang.jit_kernel.kv_cache_canary import SKIP_CHAIN_SENTINEL
from sglang.jit_kernel.kv_cache_canary_plan_ref_legacy import (
    BatchPlan,
    _build_plan,
    _translate_plan_slot_indices,
)
from sglang.srt.kv_cache_canary.host_state import translate_alive_slots_for_swa
from sglang.srt.kv_cache_canary.pool_patch import (
    PoolKind,
    attach_canary_buffers,
    get_canary_buffer_groups,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")


_FULL_POOL_SIZE: int = 16
_SWA_POOL_SIZE: int = 4


def _build_full_to_swa_lut(
    *,
    full_size: int,
    window_full_indices: list,
) -> torch.Tensor:
    """Construct a ``[full_size + 1]`` LUT mapping in-window full slots to
    consecutive swa slot indices ``[0, len(window_full_indices))`` and all
    other positions to the ``-1`` sentinel. The final entry is the LUT's
    own ``-1`` tail sentinel — sglang convention so that indexing with
    ``-1`` yields ``-1``.
    """
    lut = torch.full((full_size + 1,), -1, dtype=torch.int64)
    for swa_idx, full_idx in enumerate(window_full_indices):
        lut[full_idx] = swa_idx
    return lut


class TestTranslatePlanSlotIndices(unittest.TestCase):
    def test_in_window_slots_are_remapped_to_swa_indices(self) -> None:
        # SWA window holds full slots [10, 11, 12, 13] -> swa slots [0, 1, 2, 3].
        lut = _build_full_to_swa_lut(
            full_size=_FULL_POOL_SIZE, window_full_indices=[10, 11, 12, 13]
        )
        plan = BatchPlan(
            verify_positions=[0, 1, 2, 3],
            verify_slot_indices=[10, 11, 12, 13],
            verify_prev_slot_indices=[-1, 10, 11, 12],
            write_token_ids=[7, 8],
            write_positions=[4, 5],
            write_slot_indices=[12, 13],
            write_req_seed_slot_indices=[11],
            write_req_entry_starts=[0],
            write_req_entry_counts=[2],
            write_req_pool_indices=[1],
            num_verify=4,
            num_write=2,
            num_write_reqs=1,
        )

        translated = _translate_plan_slot_indices(plan=plan, lut=lut)

        # All in-window full slots are mapped to consecutive swa slots.
        self.assertEqual(translated.verify_slot_indices, [0, 1, 2, 3])
        # First prev-slot is chain-head sentinel -1, must pass through.
        self.assertEqual(translated.verify_prev_slot_indices, [-1, 0, 1, 2])
        self.assertEqual(translated.write_slot_indices, [2, 3])
        self.assertEqual(translated.write_req_seed_slot_indices, [1])
        # Untouched fields are preserved.
        self.assertEqual(translated.verify_positions, [0, 1, 2, 3])
        self.assertEqual(translated.write_token_ids, [7, 8])
        self.assertEqual(translated.write_positions, [4, 5])
        self.assertEqual(translated.num_verify, 4)
        self.assertEqual(translated.num_write, 2)
        self.assertEqual(translated.num_write_reqs, 1)

    def test_out_of_window_slots_map_to_minus_one(self) -> None:
        # Window covers full slots [12, 13, 14, 15]; full slot 3 is outside
        # the window and must come through as -1 rather than indexing into
        # whatever happens to live at that LUT row.
        lut = _build_full_to_swa_lut(
            full_size=_FULL_POOL_SIZE, window_full_indices=[12, 13, 14, 15]
        )
        plan = BatchPlan(
            verify_positions=[0, 1, 2, 3],
            verify_slot_indices=[3, 12, 13, 14],
            verify_prev_slot_indices=[-1, 3, 12, 13],
            write_token_ids=[],
            write_positions=[],
            write_slot_indices=[],
            write_req_seed_slot_indices=[],
            write_req_entry_starts=[],
            write_req_entry_counts=[],
            write_req_pool_indices=[],
            num_verify=4,
            num_write=0,
            num_write_reqs=0,
        )

        translated = _translate_plan_slot_indices(plan=plan, lut=lut)

        self.assertEqual(translated.verify_slot_indices, [-1, 0, 1, 2])
        # The full-slot-3 entry in prev_slot_indices also flips to -1.
        self.assertEqual(translated.verify_prev_slot_indices, [-1, -1, 0, 1])

    def test_skip_chain_sentinel_in_prev_slot_passes_through(self) -> None:
        # Sweep plans set verify_prev_slot_indices to SKIP_CHAIN_SENTINEL
        # (-2) so the kernel skips the chain hash check. The LUT translate
        # must not collapse that into the regular -1 sentinel.
        lut = _build_full_to_swa_lut(
            full_size=_FULL_POOL_SIZE, window_full_indices=[10, 11, 12, 13]
        )
        plan = BatchPlan(
            verify_positions=[0, 1],
            verify_slot_indices=[10, 11],
            verify_prev_slot_indices=[SKIP_CHAIN_SENTINEL, SKIP_CHAIN_SENTINEL],
            write_token_ids=[],
            write_positions=[],
            write_slot_indices=[],
            write_req_seed_slot_indices=[],
            write_req_entry_starts=[],
            write_req_entry_counts=[],
            write_req_pool_indices=[],
            num_verify=2,
            num_write=0,
            num_write_reqs=0,
        )

        translated = _translate_plan_slot_indices(plan=plan, lut=lut)

        self.assertEqual(translated.verify_slot_indices, [0, 1])
        self.assertEqual(
            translated.verify_prev_slot_indices,
            [SKIP_CHAIN_SENTINEL, SKIP_CHAIN_SENTINEL],
        )

    def test_seed_slot_chain_head_sentinel_passes_through(self) -> None:
        # write_req_seed_slot_indices uses -1 when K_req_old == 0 (the
        # first write into an empty req); that sentinel must survive
        # translation alongside any real seed slot in the same plan.
        lut = _build_full_to_swa_lut(
            full_size=_FULL_POOL_SIZE, window_full_indices=[10, 11, 12, 13]
        )
        plan = BatchPlan(
            verify_positions=[],
            verify_slot_indices=[],
            verify_prev_slot_indices=[],
            write_token_ids=[1, 2],
            write_positions=[0, 1],
            write_slot_indices=[10, 11],
            write_req_seed_slot_indices=[-1, 12],
            write_req_entry_starts=[0, 1],
            write_req_entry_counts=[1, 1],
            write_req_pool_indices=[1, 2],
            num_verify=0,
            num_write=2,
            num_write_reqs=2,
        )

        translated = _translate_plan_slot_indices(plan=plan, lut=lut)

        self.assertEqual(translated.write_slot_indices, [0, 1])
        self.assertEqual(translated.write_req_seed_slot_indices, [-1, 2])

    def test_full_group_bypass_when_swa_index_lut_is_none(self) -> None:
        # Regression guard: building a plan via _build_plan with no LUT
        # supplied must match the legacy un-translated behaviour exactly.
        req_to_token = torch.zeros((2, 4), dtype=torch.int32)
        req_to_token[1] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
        baseline = _build_plan(
            req_pool_indices=[1],
            seq_lens=[0],
            prefix_lens=[4],
            input_ids_list=[],
            out_cache_loc_list=[],
            positions_list=None,
            req_to_token_table=req_to_token,
            swa_window_size=None,
        )
        assert baseline is not None
        # Indices arrive unchanged because no LUT is applied; matches the
        # FULL canary group / MHA / MLA / DSV4 path.
        self.assertEqual(baseline.verify_slot_indices, [100, 101, 102, 103])
        self.assertEqual(baseline.verify_prev_slot_indices, [-1, 100, 101, 102])


class TestTranslateAliveSlotsForSWA(unittest.TestCase):
    def test_in_window_alive_slots_remap_and_out_of_window_dropped(self) -> None:
        # Alive set mixes in-window full slots (10..13) and out-of-window
        # slots (3, 5). The sweep path drops the latter rather than letting
        # the kernel dereference the swa-sized canary buffer with garbage.
        lut = _build_full_to_swa_lut(
            full_size=_FULL_POOL_SIZE, window_full_indices=[10, 11, 12, 13]
        )
        alive = torch.tensor([10, 3, 11, 5, 12, 13], dtype=torch.int64)

        translated = translate_alive_slots_for_swa(alive_slots=alive, lut=lut)

        self.assertEqual(translated.tolist(), [0, 1, 2, 3])

    def test_empty_alive_set_returns_empty(self) -> None:
        lut = _build_full_to_swa_lut(
            full_size=_FULL_POOL_SIZE, window_full_indices=[10, 11, 12, 13]
        )
        alive = torch.empty(0, dtype=torch.int64)

        translated = translate_alive_slots_for_swa(alive_slots=alive, lut=lut)

        self.assertEqual(translated.numel(), 0)


class _FakeMHASubPool:
    """SWA sub-pool stand-in (MHA-style K/V layout)."""

    def __init__(
        self, *, layer_num: int, slot_count: int, head_num: int, head_dim: int
    ) -> None:
        self.layer_num = layer_num
        self.k_buffer = [
            torch.zeros(slot_count, head_num, head_dim, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros(slot_count, head_num, head_dim, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]


class _FakeFullKVPool:
    """Full sub-pool stand-in sized larger than the swa sub-pool."""

    def __init__(
        self, *, layer_num: int, slot_count: int, head_num: int, head_dim: int
    ) -> None:
        self.layer_num = layer_num
        self.k_buffer = [
            torch.zeros(slot_count, head_num, head_dim, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros(slot_count, head_num, head_dim, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]


class _FakeRealSWAPool:
    """SWAKVPool stand-in with distinct full + swa sub-pools and a LUT."""

    def __init__(
        self,
        *,
        layer_num: int,
        full_slot_count: int,
        swa_slot_count: int,
        head_num: int,
        head_dim: int,
        lut: torch.Tensor,
        page_size: int = 1,
    ) -> None:
        self.page_size = page_size
        self.swa_page_size = page_size
        self.full_kv_pool = _FakeFullKVPool(
            layer_num=layer_num,
            slot_count=full_slot_count,
            head_num=head_num,
            head_dim=head_dim,
        )
        self.swa_kv_pool = _FakeMHASubPool(
            layer_num=layer_num,
            slot_count=swa_slot_count,
            head_num=head_num,
            head_dim=head_dim,
        )
        self.full_to_swa_index_mapping = lut

    def get_state_buf_infos(self):
        sub = self.swa_kv_pool
        k_ptrs = [b.data_ptr() for b in sub.k_buffer]
        v_ptrs = [b.data_ptr() for b in sub.v_buffer]
        k_lens = [b.nbytes for b in sub.k_buffer]
        v_lens = [b.nbytes for b in sub.v_buffer]
        k_item_lens = [b[0].nbytes * self.swa_page_size for b in sub.k_buffer]
        v_item_lens = [b[0].nbytes * self.swa_page_size for b in sub.v_buffer]
        return k_ptrs + v_ptrs, k_lens + v_lens, k_item_lens + v_item_lens


class TestSWALutCapturedOnGroup(unittest.TestCase):
    def test_real_swa_pool_captures_lut_on_swa_group_only(self) -> None:
        # Pool with a distinct full sub-pool (size 16) and swa sub-pool
        # (size 4): the SWA group must carry the LUT; the FULL group must
        # not (its canary buffer lives in full-pool index space).
        lut = _build_full_to_swa_lut(
            full_size=_FULL_POOL_SIZE, window_full_indices=[10, 11, 12, 13]
        )
        pool = _FakeRealSWAPool(
            layer_num=2,
            full_slot_count=_FULL_POOL_SIZE,
            swa_slot_count=_SWA_POOL_SIZE,
            head_num=1,
            head_dim=8,
            lut=lut,
        )
        attach_canary_buffers(pool)

        groups = get_canary_buffer_groups(pool)
        self.assertEqual(set(groups.keys()), {PoolKind.FULL, PoolKind.SWA})

        full_group = groups[PoolKind.FULL]
        swa_group = groups[PoolKind.SWA]

        # SWA canary buffer is sized off the swa sub-pool (smaller).
        self.assertEqual(swa_group.k_head.shape[0], _SWA_POOL_SIZE)
        # FULL canary buffer is sized off the full sub-pool.
        self.assertEqual(full_group.k_head.shape[0], _FULL_POOL_SIZE)

        # LUT is bound on SWA group only — the FULL group never needs it.
        self.assertIsNotNone(swa_group.swa_index_lut)
        self.assertIsNone(full_group.swa_index_lut)
        assert swa_group.swa_index_lut is not None
        self.assertEqual(
            swa_group.swa_index_lut.tolist(),
            lut.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
