"""Pool-patch tests for KV cache canary on SWA pools.

Covers ``BaseSWAKVPool``-style dispatch in ``pool_patch.attach_shadow_buffers``
plus the ``get_state_buf_infos`` patching path (PD main route for SWA).

The stateless redesign dropped SWA window-slide eviction hooks: the canary
no longer maintains host state, so there is nothing to reset when SWA
evicts a slot. The next forward simply derives K_req from sglang's own
seq_lens and the chain re-anchors on whatever slot indices the SWA pool
hands out.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.host_state import _build_plan
from sglang.srt.kv_cache_canary.pool_patch import (
    PoolKind,
    attach_shadow_buffers,
    get_shadow_groups,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


class _FakeMHASubPool:
    """Stand-in for ``swa_kv_pool`` when SWA uses an MHA-style K/V layout."""

    def __init__(
        self,
        *,
        layer_num: int,
        slot_count: int,
        head_num: int,
        head_dim: int,
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


class _FakeSWAPool:
    """Stand-in for ``SWAKVPool`` exposing the bits the canary touches."""

    def __init__(
        self,
        *,
        layer_num: int,
        slot_count: int,
        head_num: int,
        head_dim: int,
        page_size: int = 1,
    ) -> None:
        self.page_size = page_size
        self.swa_page_size = page_size
        self.swa_kv_pool = _FakeMHASubPool(
            layer_num=layer_num,
            slot_count=slot_count,
            head_num=head_num,
            head_dim=head_dim,
        )
        self.full_to_swa_index_mapping = torch.full(
            (slot_count,), -1, dtype=torch.int64
        )

    def get_state_buf_infos(self):
        sub = self.swa_kv_pool
        k_ptrs = [b.data_ptr() for b in sub.k_buffer]
        v_ptrs = [b.data_ptr() for b in sub.v_buffer]
        k_lens = [b.nbytes for b in sub.k_buffer]
        v_lens = [b.nbytes for b in sub.v_buffer]
        k_item_lens = [b[0].nbytes * self.swa_page_size for b in sub.k_buffer]
        v_item_lens = [b[0].nbytes * self.swa_page_size for b in sub.v_buffer]
        return k_ptrs + v_ptrs, k_lens + v_lens, k_item_lens + v_item_lens


class TestSWAShadowAttach(unittest.TestCase):
    def test_attach_creates_both_full_and_swa_shadow_groups(self) -> None:
        pool = _FakeSWAPool(layer_num=3, slot_count=24, head_num=1, head_dim=8)
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)

        # SWA-system pools always get TWO independent canaries: FULL + SWA.
        groups = get_shadow_groups(pool)
        self.assertEqual(set(groups.keys()), {PoolKind.FULL, PoolKind.SWA})

        # Both shadows are sized off the SWA sub-pool's slot count when the
        # fake pool has no separate full_kv_pool (DSV4-style fallback).
        self.assertEqual(groups[PoolKind.FULL].k_head.shape[0], 24)
        self.assertEqual(groups[PoolKind.SWA].k_head.shape[0], 24)
        self.assertTrue(groups[PoolKind.SWA].has_v_half)
        self.assertTrue(groups[PoolKind.FULL].has_v_half)

        # The shadow groups are independent allocations (different storage).
        self.assertNotEqual(
            groups[PoolKind.FULL].k_head.data_ptr(),
            groups[PoolKind.SWA].k_head.data_ptr(),
        )

    def test_attach_is_idempotent_on_swa_pool(self) -> None:
        pool = _FakeSWAPool(layer_num=2, slot_count=16, head_num=1, head_dim=8)
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)
        first_ptr = pool.canary_k_head.data_ptr()
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)
        self.assertEqual(pool.canary_k_head.data_ptr(), first_ptr)


class TestSWAStateBufInfosPatch(unittest.TestCase):
    def test_patched_get_state_buf_infos_preserves_kv_midpoint(self) -> None:
        pool = _FakeSWAPool(layer_num=4, slot_count=32, head_num=2, head_dim=16)
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)
        ptrs, lens, item_lens = pool.get_state_buf_infos()

        # Original [K*4, V*4] = 8 entries -> with TWO shadow groups
        # (FULL + SWA), each contributing K_head + K_tail in the K block
        # and V_head + V_tail in the V block, total becomes
        # [K*4, K_full_head, K_full_tail, K_swa_head, K_swa_tail,
        #  V*4, V_full_head, V_full_tail, V_swa_head, V_swa_tail] = 16
        # entries, and ``len // 2`` still bisects K vs V.
        self.assertEqual(len(ptrs), 16)
        self.assertEqual(len(lens), 16)
        self.assertEqual(len(item_lens), 16)
        mid = len(ptrs) // 2
        self.assertEqual(mid, 8)

        groups = get_shadow_groups(pool)
        full_group = groups[PoolKind.FULL]
        swa_group = groups[PoolKind.SWA]
        # K block tail: full first (dict order), then swa.
        self.assertEqual(ptrs[4], full_group.k_head.data_ptr())
        self.assertEqual(ptrs[5], full_group.k_tail.data_ptr())
        self.assertEqual(ptrs[6], swa_group.k_head.data_ptr())
        self.assertEqual(ptrs[7], swa_group.k_tail.data_ptr())
        # V block tail: same order.
        self.assertEqual(ptrs[12], full_group.v_head.data_ptr())
        self.assertEqual(ptrs[13], full_group.v_tail.data_ptr())
        self.assertEqual(ptrs[14], swa_group.v_head.data_ptr())
        self.assertEqual(ptrs[15], swa_group.v_tail.data_ptr())


class TestSWAVerifyWindowClipping(unittest.TestCase):
    """SWA pool: verify range must be clipped to the most recent ``swa_window_size`` slots."""

    def _build_plan_for_one_req(self, *, k_req: int, swa_window_size):
        # Single-req batch, decode step: K_req tokens written so far, no
        # new writes (n=0). Verify entries should cover the SWA window.
        # Row 0 is the padding row (req_pool_idx == 0 is skipped); row 1
        # holds this req's slot indices.
        req_to_token = torch.zeros((2, k_req), dtype=torch.int32)
        req_to_token[1] = torch.arange(1, k_req + 1, dtype=torch.int32)
        return _build_plan(
            req_pool_indices=[1],
            seq_lens=[0],
            prefix_lens=[k_req],
            input_ids_list=[],
            out_cache_loc_list=[],
            positions_list=None,
            req_to_token_table=req_to_token,
            swa_window_size=swa_window_size,
        )

    def test_swa_window_caps_verify_range_to_last_window_positions(self) -> None:
        plan = self._build_plan_for_one_req(k_req=1000, swa_window_size=128)
        assert plan is not None
        # Window-aware verify: 128 entries covering positions [872, 1000).
        self.assertEqual(plan.num_verify, 128)
        self.assertEqual(plan.verify_positions[0], 872)
        self.assertEqual(plan.verify_positions[-1], 999)
        # First entry has pos > 0 -> prev_slot_indices[0] must be the slot
        # for position (872 - 1) = 871, not -1 (the kSeed sentinel).
        expected_prev_slot = int(872)  # req_to_token[0, 871] = 871 + 1 = 872
        self.assertEqual(plan.verify_prev_slot_indices[0], expected_prev_slot)

    def test_swa_window_larger_than_k_req_uses_full_prefix(self) -> None:
        plan = self._build_plan_for_one_req(k_req=64, swa_window_size=256)
        assert plan is not None
        self.assertEqual(plan.num_verify, 64)
        self.assertEqual(plan.verify_positions[0], 0)
        # Position 0 -> prev_slot_indices = -1 (kSeed).
        self.assertEqual(plan.verify_prev_slot_indices[0], -1)

    def test_non_swa_swa_window_size_none_walks_full_prefix(self) -> None:
        plan = self._build_plan_for_one_req(k_req=10000, swa_window_size=None)
        assert plan is not None
        # No cap: every position verified (user-instruction: 10k tokens
        # decode step verifies all 10k positions).
        self.assertEqual(plan.num_verify, 10000)
        self.assertEqual(plan.verify_positions[0], 0)
        self.assertEqual(plan.verify_positions[-1], 9999)

    def test_canary_config_default_swa_window_size_is_none(self) -> None:
        cfg = CanaryConfig(mode=CanaryMode.LOG)
        self.assertIsNone(cfg.swa_window_size)


if __name__ == "__main__":
    unittest.main()
