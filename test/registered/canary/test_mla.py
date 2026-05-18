"""Host-side unit tests for KV cache canary on MLA pools (Phase 2.2).

Covers ``MLATokenToKVPool`` / ``MLATokenToKVPoolFP4`` / ``NSATokenToKVPool``
dispatch in ``pool_patch.attach_shadow_buffers``. MLA has a single latent
``kv_buffer`` per layer (no separate V half), so we only allocate a K-half
shadow and ``canary_has_v_half`` is False.

Depends on Phase 1 fix (C1/C2/C3 cuda-graph + host plan) for end-to-end
mismatch detection; only the host-side bookkeeping is exercised here.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_cache_canary.pool_patch import PoolKind, attach_shadow_buffers
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


class _FakeMLAPool:
    """Stand-in for ``MLATokenToKVPool`` exposing the bits the canary touches.

    MLA pools have a single ``kv_buffer`` (latent rep) per layer; the canary
    only allocates a K-half shadow (no V half).
    """

    def __init__(
        self,
        *,
        layer_num: int,
        slot_count: int,
        kv_cache_dim: int,
        page_size: int = 1,
    ) -> None:
        self.layer_num = layer_num
        self.page_size = page_size
        self.kv_buffer = [
            torch.zeros(slot_count, 1, kv_cache_dim, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]

    def get_contiguous_buf_infos(self):
        data_ptrs = [b.data_ptr() for b in self.kv_buffer]
        data_lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes * self.page_size for b in self.kv_buffer]
        return data_ptrs, data_lens, item_lens


class TestMLAShadowAttach(unittest.TestCase):
    def test_mla_attach_allocates_only_k_half(self) -> None:
        pool = _FakeMLAPool(layer_num=4, slot_count=32, kv_cache_dim=576)
        attach_shadow_buffers(pool, pool_kind=PoolKind.MLA)

        # MLA does NOT allocate a V half.
        self.assertIsNotNone(pool.canary_k_head)
        self.assertIsNotNone(pool.canary_k_tail)
        self.assertIsNone(pool.canary_v_head)
        self.assertIsNone(pool.canary_v_tail)
        self.assertFalse(pool.canary_has_v_half)

        # Shadow shape matches the latent representation (slot_count, 1, dim).
        self.assertEqual(pool.canary_k_head.shape, (32, 1, 576))

    def test_mla_attach_is_idempotent(self) -> None:
        pool = _FakeMLAPool(layer_num=2, slot_count=16, kv_cache_dim=128)
        attach_shadow_buffers(pool, pool_kind=PoolKind.MLA)
        first_ptr = pool.canary_k_head.data_ptr()
        attach_shadow_buffers(pool, pool_kind=PoolKind.MLA)
        self.assertEqual(pool.canary_k_head.data_ptr(), first_ptr)


class TestMLAContiguousBufInfosPatch(unittest.TestCase):
    def test_patched_appends_k_head_and_k_tail(self) -> None:
        pool = _FakeMLAPool(layer_num=4, slot_count=32, kv_cache_dim=576)
        attach_shadow_buffers(pool, pool_kind=PoolKind.MLA)
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()

        # Original 4 entries (one per layer), +2 canary entries = 6 total.
        self.assertEqual(len(ptrs), 6)
        self.assertEqual(len(lens), 6)
        self.assertEqual(len(item_lens), 6)
        # Canary entries appended at the tail; MLA has a single kv_buffer
        # series so there is no K|V midpoint to preserve here.
        self.assertEqual(ptrs[4], pool.canary_k_head.data_ptr())
        self.assertEqual(ptrs[5], pool.canary_k_tail.data_ptr())
        # Per-slot stride is uniform across the original layers and the canary
        # entries (canary slot stride = real layer slot stride).
        layer0_item_len = item_lens[0]
        self.assertEqual(item_lens[4], layer0_item_len)
        self.assertEqual(item_lens[5], layer0_item_len)

    def test_slot_stride_bytes_matches_latent_dim(self) -> None:
        pool = _FakeMLAPool(layer_num=2, slot_count=8, kv_cache_dim=512)
        attach_shadow_buffers(pool, pool_kind=PoolKind.MLA)
        # slot_stride_bytes = nbytes of one slot in the latent representation:
        # 1 head * 512 dim * 2 bytes (bf16) = 1024.
        expected = 1 * 512 * 2
        self.assertEqual(pool.canary_slot_stride_bytes, expected)


class TestMLADispatchFallthroughForNSAAndFP4(unittest.TestCase):
    """NSATokenToKVPool and MLATokenToKVPoolFP4 both inherit MLATokenToKVPool.

    They expose the same ``kv_buffer`` attribute (no ``k_buffer``/``v_buffer``)
    and same ``get_contiguous_buf_infos`` shape — the duck-typed dispatch in
    ``attach_shadow_buffers`` routes them through ``_attach_mla`` exactly like
    the base MLA pool. This test mimics the NSA case by also exposing an
    auxiliary buffer that the canary should ignore.
    """

    def test_nsa_like_pool_routes_through_mla_branch(self) -> None:
        pool = _FakeMLAPool(layer_num=3, slot_count=16, kv_cache_dim=576)
        # Mimic NSA's auxiliary buffer; the canary must NOT shadow it.
        pool.index_k_with_scale_buffer = [
            torch.zeros(4, 256, dtype=torch.uint8) for _ in range(3)
        ]
        attach_shadow_buffers(pool, pool_kind=PoolKind.MLA)

        self.assertFalse(pool.canary_has_v_half)
        # Index buffer is unchanged; canary doesn't touch it.
        self.assertEqual(len(pool.index_k_with_scale_buffer), 3)
        ptrs, _, _ = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 3 + 2)


if __name__ == "__main__":
    unittest.main()
