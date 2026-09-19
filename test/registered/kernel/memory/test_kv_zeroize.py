"""KV zeroization: the bytes behind a freed slot are really erased.

The wipe is a Triton kernel over a pool's real buffers, so the only test worth
writing reads the bytes back.

    python -m pytest test/registered/unit/mem_cache/test_kv_zeroize.py -v
"""

import unittest
from array import array

import torch

from sglang.kernels.ops.kvcache.zero_kv_rows import zero_kv_rows
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_zeroize import (
    KvZeroizeUnsupported,
    build_kv_zeroize_plan,
    build_kv_zeroizer,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_DTYPE = torch.bfloat16
_LAYERS = 3
_HEADS = 2
_HEAD_DIM = 16
# Distinct from _HEAD_DIM so k and v rows differ in size within one plan.
_V_HEAD_DIM = 8
_MARKER = 0x5A


def _make_mha_pool(size: int, page_size: int) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=_DTYPE,
        head_num=_HEADS,
        head_dim=_HEAD_DIM,
        layer_num=_LAYERS,
        device="cuda",
        enable_memory_saver=False,
        v_head_dim=_V_HEAD_DIM,
    )


def _stamp(pool: MHATokenToKVPool) -> list[torch.Tensor]:
    """Fill every KV byte with the marker; return the per-buffer byte views."""
    views = [buf.view(torch.uint8) for buf in (*pool.k_buffer, *pool.v_buffer)]
    for view in views:
        view.fill_(_MARKER)
    return views


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestKvZeroize(CustomTestCase):
    def test_only_the_requested_rows_are_cleared_in_every_buffer(self):
        pool = _make_mha_pool(size=64, page_size=1)
        views = _stamp(pool)
        plan = build_kv_zeroize_plan(pool)
        # k and v rows really do differ, else the per-buffer stride is untested.
        k_words = _HEADS * _HEAD_DIM * _DTYPE.itemsize // 8
        v_words = _HEADS * _V_HEAD_DIM * _DTYPE.itemsize // 8
        self.assertEqual(sorted(set(plan.row_words.tolist())), [v_words, k_words])

        doomed = torch.tensor([3, 4, 9], device="cuda", dtype=torch.int64)
        plan.zeroize(doomed, page_aligned_run=True)
        torch.cuda.synchronize()

        kept = [
            row for row in range(pool.size + pool.page_size) if row not in (3, 4, 9)
        ]
        for view in views:
            self.assertTrue(bool((view[doomed] == 0).all()))
            self.assertTrue(bool((view[kept] == _MARKER).all()))

    def test_a_page_row_pool_clears_whole_pages(self):
        page_size = 4
        # HND is the layout whose leading dim counts pages rather than slots.
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            pool = _make_mha_pool(size=60, page_size=page_size)
        self.assertEqual(
            pool.k_buffer[0].shape[0], (pool.size + page_size) // page_size
        )
        views = _stamp(pool)
        plan = build_kv_zeroize_plan(pool)
        self.assertEqual(plan.tokens_per_row, page_size)
        # Per token: every layer's k and v row, divided over the page's tokens.
        self.assertEqual(
            plan.bytes_per_token,
            _LAYERS * _HEADS * (_HEAD_DIM + _V_HEAD_DIM) * _DTYPE.itemsize,
        )

        # Slots 8..15, i.e. whole pages 2 and 3.
        plan.zeroize(torch.arange(8, 16, device="cuda"), page_aligned_run=True)
        torch.cuda.synchronize()

        for view in views:
            self.assertTrue(bool((view[[2, 3]] == 0).all()))
            self.assertTrue(bool((view[[1, 4]] == _MARKER).all()))

    def test_the_kernel_masks_each_buffer_against_its_own_row(self):
        # The grid is sized for the widest row in the table, so a narrow buffer
        # is over-covered; masking on the table-wide width would run each row's
        # stores into the next rows of that buffer.
        wide = torch.full((4, 300), -1, dtype=torch.int64, device="cuda")
        narrow = torch.full((4, 3), -1, dtype=torch.int64, device="cuda")
        zero_kv_rows(
            base_ptrs=torch.tensor(
                [wide.data_ptr(), narrow.data_ptr()], dtype=torch.int64, device="cuda"
            ),
            row_words=torch.tensor([300, 3], dtype=torch.int64, device="cuda"),
            row_ids=torch.tensor([1], dtype=torch.int64, device="cuda"),
            max_row_words=300,
        )
        torch.cuda.synchronize()

        for buf in (wide, narrow):
            self.assertTrue(bool((buf[1] == 0).all()))
            self.assertTrue(bool((buf[[0, 2, 3]] == -1).all()))

    def test_pools_whose_bytes_cannot_be_enumerated_are_rejected(self):
        # A partial wipe reads as a guarantee, so an un-enumerable pool must
        # fail at build time rather than clear only the buffers it understands.
        def scales_outside_kv():
            # FP4/MXFP8 keep per-block scales in their own allocations.
            pool = _make_mha_pool(size=16, page_size=1)
            pool.k_scale_buffer = [torch.zeros(1, device="cuda")]
            return pool

        def aliased_layers():
            pool = _make_mha_pool(size=16, page_size=1)
            pool.k_buffer = [pool.k_buffer[0]] * _LAYERS
            return pool

        def mla_subclass_with_its_own_scales():
            # MLATokenToKVPoolFP4 passes an isinstance(MLATokenToKVPool) test
            # but keeps kv_scale_buffer outside kv_buffer.
            return MLATokenToKVPoolFP4(
                size=16,
                page_size=1,
                dtype=torch.float8_e4m3fn,
                kv_lora_rank=32,
                qk_rope_head_dim=16,
                layer_num=2,
                device="cuda",
                enable_memory_saver=False,
            )

        for make_pool, message in (
            (scales_outside_kv, "per-block scales"),
            (aliased_layers, "alias one storage"),
            (mla_subclass_with_its_own_scales, "MLATokenToKVPoolFP4 is not supported"),
        ):
            with self.subTest(message):
                with self.assertRaisesRegex(KvZeroizeUnsupported, message):
                    build_kv_zeroize_plan(make_pool())

    def test_an_mla_pool_clears_the_latent_row_that_holds_both_k_and_v(self):
        pool = MLATokenToKVPool(
            size=32,
            page_size=1,
            dtype=_DTYPE,
            kv_lora_rank=32,
            qk_rope_head_dim=16,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        views = [buf.view(torch.uint8) for buf in pool.kv_buffer]
        for view in views:
            view.fill_(_MARKER)

        build_kv_zeroize_plan(pool).zeroize(
            torch.tensor([5], device="cuda"), page_aligned_run=True
        )
        torch.cuda.synchronize()

        for layer in range(2):
            # V is kv_buffer[..., :kv_lora_rank]; clearing the row clears both.
            self.assertTrue(bool((pool.get_key_buffer(layer)[5] == 0).all()))
            self.assertTrue(bool((pool.get_value_buffer(layer)[5] == 0).all()))
            self.assertTrue(bool((views[layer][[4, 6]] == _MARKER).all()))

    def test_swa_slots_are_resolved_before_anything_is_freed(self):
        # The SWA component frees the node's FULL slot ids and the allocator
        # translates them through full_to_swa_index_mapping. If the FULL side
        # were freed first, a realloc would rebind that mapping and the wipe
        # would land on a live tenant's SWA slot -- so the translation happens
        # when the slots are queued, not when they are cleared.
        pool = SWAKVPool(
            size=64,
            size_swa=32,
            page_size=1,
            dtype=_DTYPE,
            head_num=_HEADS,
            head_dim=_HEAD_DIM,
            swa_attention_layer_ids=[0, 1],
            full_attention_layer_ids=[2],
            device="cuda",
            enable_memory_saver=False,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=32,
            page_size=1,
            dtype=_DTYPE,
            device="cuda",
            kvcache=pool,
            need_sort=False,
        )
        zeroizer = build_kv_zeroizer(
            allocator=allocator,
            page_size=1,
            components=(ComponentType.FULL, ComponentType.SWA),
        )
        full = torch.tensor([5, 6], device="cuda", dtype=torch.int64)
        swa = torch.tensor([11, 12], device="cuda", dtype=torch.int64)
        allocator.set_full_to_swa_mapping(full, swa)
        for layer in range(2):
            pool.swa_kv_pool.k_buffer[layer].fill_(1.0)
            pool.swa_kv_pool.v_buffer[layer].fill_(1.0)

        resolved = zeroizer.prepare(ComponentType.SWA, full)
        self.assertEqual(sorted(resolved.tolist()), [11, 12])

        # A later rebind (what a realloc of the freed FULL slot would do) must
        # not move the wipe onto the new SWA slots.
        allocator.set_full_to_swa_mapping(
            full, torch.tensor([20, 21], device="cuda", dtype=torch.int64)
        )
        zeroizer.zeroize(ComponentType.SWA, resolved, page_aligned_run=False)
        torch.cuda.synchronize()

        for layer in range(2):
            self.assertTrue(bool((pool.swa_kv_pool.k_buffer[layer][swa] == 0).all()))
            self.assertTrue(
                bool((pool.swa_kv_pool.k_buffer[layer][[20, 21]] != 0).all())
            )


def _make_cache_with_zeroize(pool, budget_bytes: int) -> UnifiedRadixCache:
    allocator = TokenToKVPoolAllocator(
        size=pool.size,
        dtype=_DTYPE,
        device="cuda",
        kvcache=pool,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=128, device="cuda", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
    )
    cache.enable_cache_salt_ttl_zeroize(
        build_kv_zeroizer(allocator, 1, cache.components.keys()), budget_bytes
    )
    return cache


def _insert(cache, token_ids, cache_salt) -> torch.Tensor:
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids), cache_salt=cache_salt),
            value=indices.to(torch.int64),
        )
    )
    return indices.to(torch.int64)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestCacheSaltTtlZeroize(CustomTestCase):
    """The TTL wiring: expiry clears the bytes before the slots are reusable."""

    def test_expiry_clears_the_bytes_and_returns_the_slots(self):
        pool = _make_mha_pool(size=64, page_size=1)
        cache = _make_cache_with_zeroize(pool, budget_bytes=1 << 30)
        doomed = _insert(cache, [1, 2, 3], "salt-a")
        kept = _insert(cache, [4, 5, 6], "salt-b")
        views = _stamp(pool)
        available_before = cache.token_to_kv_pool_allocator.available_size()

        cache.expire_cache_salts(["salt-a"])
        torch.cuda.synchronize()

        for view in views:
            self.assertTrue(bool((view[doomed] == 0).all()))
            self.assertTrue(bool((view[kept] == _MARKER).all()))
        self.assertEqual(
            cache.token_to_kv_pool_allocator.available_size(),
            available_before + len(doomed),
        )
        self.assertFalse(cache.has_pending_cache_salt_expiry())

    def test_a_slot_the_budget_did_not_reach_is_withheld_not_handed_out_dirty(self):
        pool = _make_mha_pool(size=64, page_size=1)
        # Two tokens' worth per iteration, so a 6-token prefix takes three.
        bytes_per_token = _LAYERS * _HEADS * (_HEAD_DIM + _V_HEAD_DIM) * _DTYPE.itemsize
        cache = _make_cache_with_zeroize(pool, budget_bytes=2 * bytes_per_token)
        doomed = _insert(cache, [1, 2, 3, 4, 5, 6], "salt-a")
        views = _stamp(pool)
        available_before = cache.token_to_kv_pool_allocator.available_size()

        cache.expire_cache_salts(["salt-a"])
        torch.cuda.synchronize()

        # Only the budgeted prefix is cleared, and only it is freed: an
        # uncleared slot stays allocated rather than going back dirty.
        for view in views:
            self.assertTrue(bool((view[doomed[:2]] == 0).all()))
            self.assertTrue(bool((view[doomed[2:]] == _MARKER).all()))
        self.assertEqual(
            cache.token_to_kv_pool_allocator.available_size(), available_before + 2
        )
        self.assertTrue(cache.has_pending_cache_salt_expiry())

        for _ in range(2):
            cache.drain_expiring_cache_salts()
        torch.cuda.synchronize()

        for view in views:
            self.assertTrue(bool((view[doomed] == 0).all()))
        self.assertEqual(
            cache.token_to_kv_pool_allocator.available_size(),
            available_before + len(doomed),
        )
        self.assertFalse(cache.has_pending_cache_salt_expiry())


if __name__ == "__main__":
    unittest.main()
