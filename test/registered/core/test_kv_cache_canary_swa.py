"""Host-side unit tests for KV cache canary on SWA pools (Phase 2.1).

Covers ``BaseSWAKVPool``-style dispatch in ``pool_patch.attach_shadow_buffers``
plus the ``get_state_buf_infos`` patching path (PD main route for SWA) and the
window-slide eviction hook.

Depends on Phase 1 fix (C1/C2/C3 cuda-graph + host plan) for end-to-end
mismatch detection; only the host-side bookkeeping is exercised here.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.host_state import CanaryHostState
from sglang.srt.kv_cache_canary.pool_patch import (
    PoolKind,
    attach_shadow_buffers,
    install_swa_free_hook,
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
        # Mimics ``BaseSWAKVPool.get_state_buf_infos``: returns the SWA
        # sub-pool's contiguous K|V layout.
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

    def free_swa(self, free_index: torch.Tensor) -> None:
        # No-op for the test stub; the hook only cares that ``free_swa`` is
        # called and the on-free callback fires.
        return


class TestSWAShadowAttach(unittest.TestCase):
    def test_attach_uses_swa_sub_pool_slot_index_space(self) -> None:
        pool = _FakeSWAPool(layer_num=3, slot_count=24, head_num=1, head_dim=8)
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)

        # The shadow tensors are sized off the SWA sub-pool (24 slots), NOT
        # any larger full-pool slot space. This is the invariant: SWA must
        # never reuse the full pool's slot indices.
        self.assertEqual(pool.canary_k_head.shape[0], 24)
        self.assertEqual(pool.canary_v_head.shape[0], 24)
        self.assertTrue(pool.canary_has_v_half)

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

        # Original layout: [K_l0..K_l3, V_l0..V_l3] = 8 entries.
        # Patched: [K_l0..K_l3, K_head, K_tail, V_l0..V_l3, V_head, V_tail] = 12,
        # so ``len // 2`` still bisects K vs V (6 vs 6) — what downstream PD
        # code relies on.
        self.assertEqual(len(ptrs), 12)
        self.assertEqual(len(lens), 12)
        self.assertEqual(len(item_lens), 12)
        mid = len(ptrs) // 2
        self.assertEqual(mid, 6)
        self.assertEqual(ptrs[4], pool.canary_k_head.data_ptr())
        self.assertEqual(ptrs[5], pool.canary_k_tail.data_ptr())
        self.assertEqual(ptrs[10], pool.canary_v_head.data_ptr())
        self.assertEqual(ptrs[11], pool.canary_v_tail.data_ptr())


class TestSWAEvictionHook(unittest.TestCase):
    def test_free_swa_triggers_on_free_callback(self) -> None:
        pool = _FakeSWAPool(layer_num=2, slot_count=8, head_num=1, head_dim=4)
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)
        calls = []
        install_swa_free_hook(pool=pool, on_free=lambda: calls.append("evict"))
        pool.free_swa(torch.tensor([0, 1, 2], dtype=torch.int64))
        pool.free_swa(torch.tensor([3], dtype=torch.int64))
        self.assertEqual(calls, ["evict", "evict"])

    def test_free_swa_hook_is_idempotent_per_pool(self) -> None:
        pool = _FakeSWAPool(layer_num=1, slot_count=4, head_num=1, head_dim=2)
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)
        calls_a = []
        calls_b = []
        install_swa_free_hook(pool=pool, on_free=lambda: calls_a.append("a"))
        # Second hook install on the same pool is a no-op; only the first
        # callback stays attached.
        install_swa_free_hook(pool=pool, on_free=lambda: calls_b.append("b"))
        pool.free_swa(torch.tensor([0], dtype=torch.int64))
        self.assertEqual(calls_a, ["a"])
        self.assertEqual(calls_b, [])


class TestSWAHostStateEvictionReset(unittest.TestCase):
    def _make_state(self) -> CanaryHostState:
        config = CanaryConfig(mode=CanaryMode.LOG)
        return CanaryHostState(config=config, num_req_slots=64)

    def test_reset_all_last_committed_clears_only_last_committed_keeps_k_req(
        self,
    ) -> None:
        state = self._make_state()
        plan = state.plan_batch(
            req_pool_indices=[1, 2],
            req_token_counts=[2, 1],
            req_start_positions=[0, 0],
            input_tokens_per_req=[[11, 22], [33]],
            write_slot_indices_per_req=[[100, 101], [200]],
        )
        state.commit_plan(plan)
        # Sanity: both reqs now have a last_committed.
        self.assertTrue(state.has_state(1))
        self.assertTrue(state.has_state(2))

        state.reset_all_last_committed()
        # last_committed gone but the request state itself is still tracked
        # (k_req must survive so future writes resume at the correct position).
        # Build a follow-up plan and confirm: zero verify entries, the writes
        # advance from the prior k_req.
        plan2 = state.plan_batch(
            req_pool_indices=[1],
            req_token_counts=[1],
            req_start_positions=[2],
            input_tokens_per_req=[[55]],
            write_slot_indices_per_req=[[102]],
        )
        self.assertEqual(plan2.num_verify, 0)
        self.assertEqual(plan2.num_write, 1)


if __name__ == "__main__":
    unittest.main()
