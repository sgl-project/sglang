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

from sglang.srt.kv_cache_canary.pool_patch import PoolKind, attach_shadow_buffers
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
    def test_attach_uses_swa_sub_pool_slot_index_space(self) -> None:
        pool = _FakeSWAPool(layer_num=3, slot_count=24, head_num=1, head_dim=8)
        attach_shadow_buffers(pool, pool_kind=PoolKind.SWA)

        # Shadows sized off the SWA sub-pool (24 slots), NOT any larger
        # full-pool slot space — SWA must never reuse full-pool slot indices.
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

        # Original [K*4, V*4] = 8 entries -> patched [K*4, K_head, K_tail,
        # V*4, V_head, V_tail] = 12, so ``len // 2`` still bisects K vs V.
        self.assertEqual(len(ptrs), 12)
        self.assertEqual(len(lens), 12)
        self.assertEqual(len(item_lens), 12)
        mid = len(ptrs) // 2
        self.assertEqual(mid, 6)
        self.assertEqual(ptrs[4], pool.canary_k_head.data_ptr())
        self.assertEqual(ptrs[5], pool.canary_k_tail.data_ptr())
        self.assertEqual(ptrs[10], pool.canary_v_head.data_ptr())
        self.assertEqual(ptrs[11], pool.canary_v_tail.data_ptr())


if __name__ == "__main__":
    unittest.main()
