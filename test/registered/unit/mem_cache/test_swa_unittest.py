import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored
from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import (
    available_and_evictable_str,
    free_kv_row_segments,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _event_hashes(events):
    return [block_hash for event in events for block_hash in event.block_hashes]


class _DummyReq:
    def __init__(self):
        self._kv_committed_len = 0
        self.swa_prefix_lock_released = False
        self.kv = SimpleNamespace(swa_evicted_seqlen=0, cache_protected_len=0)


def _build_swa_tree(
    is_eagle: bool,
    page_size: int = 1,
    req_size: int = 8,
    max_context_len: int = 64,
    kv_size: int = 64,
    kv_size_swa: int = 32,
    sliding_window_size: int = 4,
    enable_kv_cache_events: bool = False,
):
    head_num = 8
    head_dim = 128
    num_layers = 24
    global_interval = 4
    dtype = torch.bfloat16
    device = get_device()
    full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
    full_attention_layer_ids_set = set(full_attention_layer_ids)
    swa_attention_layer_ids = [
        i for i in range(num_layers) if i not in full_attention_layer_ids_set
    ]

    req_to_token_pool = ReqToTokenPool(
        size=req_size,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        device=device,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = SWARadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            is_eagle=is_eagle,
            sliding_window_size=sliding_window_size,
            enable_kv_cache_events=enable_kv_cache_events,
        ),
    )
    return tree, allocator, req_to_token_pool


def _sync_error(fn):
    """The RuntimeError torch raises if `fn` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


def _build_pure_swa_allocator(size_swa: int = 16):
    device = get_device()
    kv_pool = SWAKVPool(
        size=0,
        size_swa=size_swa,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=8,
        head_dim=128,
        swa_attention_layer_ids=list(range(4)),
        full_attention_layer_ids=[],
        device=device,
    )
    return PureSWATokenToKVPoolAllocator(
        size_swa=size_swa,
        page_size=1,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )


def _swa_alloc(allocator, need_size):
    """SWA-pool alloc that also works for page_size > 1 (built-in alloc asserts page_size == 1)."""
    if allocator.page_size == 1:
        return allocator.alloc(need_size)

    assert need_size % allocator.page_size == 0
    full_indices = allocator.full_attn_allocator.alloc(need_size)
    swa_indices = allocator.swa_attn_allocator.alloc(need_size)
    assert full_indices is not None and swa_indices is not None
    allocator.full_to_swa_index_mapping[full_indices] = swa_indices
    return full_indices


def _insert(tree, allocator, token_ids):
    indices = _swa_alloc(allocator, len(token_ids))
    assert indices is not None
    tree.insert(InsertParams(key=RadixKey(array("q", token_ids)), value=indices))


def _insert_chain(tree, allocator, token_ids):
    _insert(tree, allocator, token_ids)
    match = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", token_ids))))
    return match.last_device_node


def _expected_tail_size(window: int, page_size: int) -> int:
    """Mirror of _maybe_split_leaf_for_swa_lock's tail_size formula."""
    return (window + page_size - 1) // page_size * page_size


class TestSWA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_swa_radix_cache_kv_events(self):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False, enable_kv_cache_events=True
        )
        tree.take_events()  # Clear the reset event.

        _insert(tree, allocator, [1, 2, 3, 4])
        first_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(first_insert_events), 1)
        self.assertEqual(list(first_insert_events[0].token_ids), [1, 2, 3, 4])

        _insert(tree, allocator, [1, 2, 3, 4, 5, 6])
        second_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(second_insert_events), 1)
        self.assertEqual(list(second_insert_events[0].token_ids), [5, 6])

        stored_hashes = [
            block_hash
            for event in first_insert_events + second_insert_events
            for block_hash in event.block_hashes
        ]

        # Evicting only SWA tokens tombstones nodes but keeps full KV blocks.
        result = tree.evict(EvictParams(num_tokens=0, swa_num_tokens=1))
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertGreaterEqual(result.swa_num_tokens_evicted, 1)
        self.assertEqual(
            [e for e in tree.take_events() if isinstance(e, BlockRemoved)], []
        )

        result = tree.evict(EvictParams(num_tokens=1, swa_num_tokens=0))
        self.assertGreaterEqual(result.num_tokens_evicted, 1)
        removed_hashes = _event_hashes(
            [e for e in tree.take_events() if isinstance(e, BlockRemoved)]
        )
        self.assertCountEqual(removed_hashes, stored_hashes)

    def test_swa_radix_cache_kv_events_split_hash(self):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False, enable_kv_cache_events=True
        )
        tree.take_events()  # Clear the reset event.

        _insert(tree, allocator, [1, 2, 3, 4])
        first_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(first_insert_events), 1)
        split_parent_hash = first_insert_events[0].block_hashes[1]

        _insert(tree, allocator, [1, 2, 5, 6])
        second_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(second_insert_events), 1)
        self.assertEqual(list(second_insert_events[0].token_ids), [5, 6])
        self.assertEqual(second_insert_events[0].parent_block_hash, split_parent_hash)

    def test_swa_memory_pool_paged_free_clears_full_page_mapping(self):
        page_size = 4
        _, allocator, _ = _build_swa_tree(
            is_eagle=False,
            page_size=page_size,
            kv_size=16,
            kv_size_swa=16,
            sliding_window_size=page_size,
        )

        full_indices = _swa_alloc(allocator, page_size)
        self.assertEqual(allocator.swa_available_size(), 16 - page_size)

        allocator.free_swa(full_indices[:1])
        self.assertEqual(allocator.swa_available_size(), 16)
        self.assertTrue(
            torch.all(
                allocator.full_to_swa_index_mapping[full_indices.to(torch.int64)] == 0
            )
        )

        allocator.free_swa(full_indices[1:2])
        self.assertEqual(allocator.swa_available_size(), 16)

    @unittest.skipUnless(torch.cuda.is_available(), "sync detection needs CUDA")
    def test_clearing_the_mapping_does_not_synchronize(self):
        """Clearing the full-to-SWA mapping must not block the stream; writing a
        host-resident scalar into it does.
        """
        _, allocator, _ = _build_swa_tree(is_eagle=False)
        full_indices = _swa_alloc(allocator, 4)
        mapping = allocator.full_to_swa_index_mapping

        # Warm up outside the window: a first-time cudaMalloc can synchronize on
        # its own, which the detector would report as this call's fault.
        allocator.clear_full_to_swa_mapping(full_indices)

        # Gate on the pre-fix form: a detector blind to this sync class would pass
        # the assert below no matter how the mapping is cleared.
        pre_fix_error = _sync_error(
            lambda: mapping.__setitem__(full_indices.to(torch.int64), 0)
        )
        if pre_fix_error is None:
            self.skipTest("sync debug mode does not flag a blocking H2D copy here")

        self.assertIsNone(
            _sync_error(lambda: allocator.clear_full_to_swa_mapping(full_indices))
        )

    def test_free_swa_group_owns_deferred_indices(self):
        _, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=32,
            kv_size_swa=32,
        )
        index_batches = []
        for size in (2, 3, 1, 4):
            indices = _swa_alloc(allocator, size)
            assert indices is not None
            index_batches.append(indices)
        original_indices = torch.cat([indices.clone() for indices in index_batches])

        available_before_free = allocator.swa_available_size()
        allocator.free_group_begin()
        for indices in index_batches:
            allocator.free_swa(indices)

        self.assertEqual(len(allocator.swa_free_group), len(index_batches))
        self.assertEqual(allocator.swa_available_size(), available_before_free)
        for indices in index_batches:
            indices.zero_()
        allocator.free_group_end()

        self.assertTrue(
            torch.equal(
                allocator.full_to_swa_index_mapping[original_indices.to(torch.int64)],
                torch.zeros_like(original_indices),
            )
        )
        self.assertEqual(
            allocator.swa_available_size(),
            available_before_free + original_indices.numel(),
        )

    def test_free_swa_group_owns_mapping_at_enqueue_time(self):
        _, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=8,
            kv_size_swa=8,
        )
        old_full = _swa_alloc(allocator, 1)
        new_full = _swa_alloc(allocator, 1)
        assert old_full is not None and new_full is not None
        old_swa = allocator.full_to_swa_index_mapping[old_full].clone()
        new_swa = allocator.full_to_swa_index_mapping[new_full].clone()

        allocator.free_group_begin()
        allocator.free_swa(old_full)

        # Cache reconciliation can transfer a different SWA slot onto the same
        # full slot before the group flushes. The deferred free still owns the
        # mapping observed above, not this replacement mapping.
        allocator.set_full_to_swa_mapping(old_full, new_swa)
        allocator.clear_full_to_swa_mapping(new_full)
        allocator.free_group_end()

        torch.testing.assert_close(
            allocator.full_to_swa_index_mapping[old_full], new_swa
        )
        self.assertTrue(
            torch.isin(old_swa, allocator.swa_attn_allocator.free_pages).item()
        )
        self.assertFalse(
            torch.isin(new_swa, allocator.swa_attn_allocator.free_pages).item()
        )

    def _build_two_mapped_slots(self, page_size=1):
        _, allocator, _ = _build_swa_tree(
            is_eagle=False,
            page_size=page_size,
            kv_size=8 * page_size,
            kv_size_swa=8 * page_size,
        )
        old_full = _swa_alloc(allocator, page_size)
        new_full = _swa_alloc(allocator, page_size)
        assert old_full is not None and new_full is not None
        old_swa = allocator.full_to_swa_index_mapping[old_full].clone()
        new_swa = allocator.full_to_swa_index_mapping[new_full].clone()
        return allocator, old_full, new_full, old_swa, new_swa

    def _swa_slot_is_free(self, allocator, swa_index):
        # free_pages holds page ids for page_size > 1 and token ids otherwise,
        # so compare in page space (a no-op divide when page_size == 1).
        swa_pages = swa_index // allocator.page_size
        free_pages = allocator.swa_attn_allocator.free_pages
        return bool(torch.isin(swa_pages, free_pages).all().item())

    def _run_remap_during_free_group(self, allocator, old_full, new_full, new_swa):
        """Queue a combined free, then transfer another SWA slot onto the same
        full slot before the group flushes -- what tombstone recovery does."""
        allocator.free_group_begin()
        allocator.free(old_full)
        allocator.set_full_to_swa_mapping(old_full, new_swa)
        allocator.clear_full_to_swa_mapping(new_full)
        allocator.free_group_end()

    def test_free_group_owns_mapping_at_enqueue_time(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                allocator, old_full, new_full, old_swa, new_swa = (
                    self._build_two_mapped_slots(page_size=page_size)
                )
                available_before = allocator.swa_available_size()

                self._run_remap_during_free_group(
                    allocator, old_full, new_full, new_swa
                )

                self.assertTrue(
                    self._swa_slot_is_free(allocator, old_swa),
                    "the SWA slot owned at enqueue time leaked",
                )
                self.assertFalse(
                    self._swa_slot_is_free(allocator, new_swa),
                    "the replacement SWA slot was freed while still mapped",
                )
                self.assertEqual(
                    allocator.swa_available_size(), available_before + page_size
                )
                # Everything still in use stays reachable through the mapping.
                mapped = allocator.full_to_swa_index_mapping[:-1]
                num_mapped = int((mapped > 0).sum().item())
                num_in_use = (
                    allocator.swa_attn_allocator.size - allocator.swa_available_size()
                )
                self.assertEqual(num_mapped, num_in_use)

    def test_pure_swa_rejects_mapping_edits(self):
        allocator = _build_pure_swa_allocator()
        indices = allocator.alloc(2)
        with self.assertRaises(NotImplementedError):
            allocator.clear_full_to_swa_mapping(indices)
        with self.assertRaises(NotImplementedError):
            allocator.set_full_to_swa_mapping(indices, indices)
        torch.testing.assert_close(
            allocator.full_to_swa_index_mapping[indices], indices
        )

    def test_swa_radix_cache_1(self):
        # args
        req_size = 10
        max_context_len = 128
        kv_size = 128
        kv_size_swa = 64
        page_size = 1
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        # setup req to token pool
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # setup kv pool
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
        )
        # setup token to kv pool allocator
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                disable=False,
                page_size=page_size,
                sliding_window_size=sliding_window_size,
            ),
        )

        # test
        print(
            f"[Start] allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        self.assertEqual(len(req1_token_ids), len(req1_kv_indices))
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        key = RadixKey(array("q", req1_token_ids))
        result = tree.insert(InsertParams(key=key, value=req1_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        key = RadixKey(array("q", req2_token_ids))
        result = tree.insert(InsertParams(key=key, value=req2_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        key = RadixKey(array("q", req3_token_ids))
        result = tree.insert(InsertParams(key=key, value=req3_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        key = RadixKey(array("q", req4_token_ids))
        result = tree.insert(InsertParams(key=key, value=req4_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", req5_token_ids)))
        )
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", req6_token_ids)))
        )
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 7)
        self.assertEqual(len(last_node.key), 2)
        self.assertEqual(last_node.key.token_ids[0], 60)
        self.assertEqual(last_node.key.token_ids[1], 70)

        print(tree.available_and_evictable_str())
        print(available_and_evictable_str(tree))
        tree.sanity_check()

    def test_swa_radix_cache_eagle(self):
        # args
        req_size = 10
        max_context_len = 128
        kv_size = 128
        kv_size_swa = 64
        page_size = 1
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        # setup req to token pool
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # setup kv pool
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
        )
        # setup token to kv pool allocator
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                disable=False,
                is_eagle=True,
                sliding_window_size=sliding_window_size,
            ),
        )

        # test
        print(
            f"[Start] allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        self.assertEqual(len(req1_token_ids), len(req1_kv_indices))
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        key = RadixKey(array("q", req1_token_ids))
        result = tree.insert(InsertParams(key=key, value=req1_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 0)
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        key = RadixKey(array("q", req2_token_ids))
        result = tree.insert(InsertParams(key=key, value=req2_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 2)
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        key = RadixKey(array("q", req3_token_ids))
        result = tree.insert(InsertParams(key=key, value=req3_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 0)
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        key = RadixKey(array("q", req4_token_ids))
        result = tree.insert(InsertParams(key=key, value=req4_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 4)
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.num_tokens_evicted >= full_num_tokens
        )  # May evict more due to node granularity
        print(
            f"evicted {evict_result.num_tokens_evicted} full tokens, {evict_result.swa_num_tokens_evicted} swa tokens"
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.swa_num_tokens_evicted >= swa_num_tokens
        ), f"evicted {evict_result.swa_num_tokens_evicted} swa tokens, expected {swa_num_tokens}"
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.num_tokens_evicted >= full_num_tokens
        ), f"evicted {evict_result.num_tokens_evicted} full tokens, expected {full_num_tokens}"
        assert (
            evict_result.swa_num_tokens_evicted >= swa_num_tokens
        ), f"evicted {evict_result.swa_num_tokens_evicted} swa tokens, expected {swa_num_tokens}"
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", req5_token_ids)))
        )
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)  # no swa prefix matched

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", req6_token_ids)))
        )
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 6)
        self.assertEqual(len(last_node.key), 2)
        # Bigram view: token_ids holds raw tokens; iteration yields bigram tuples.
        self.assertTrue(last_node.key.is_bigram)
        self.assertEqual(list(last_node.key), [(5, 60), (60, 70)])

    def test_swa_cache_finished_req_eagle_uses_cache_protected_len_and_bigram_key(self):
        tree, allocator, req_to_token_pool = _build_swa_tree(is_eagle=True)

        # Case 1: is_insert=True should pass bigram key and use cache_protected_len.
        req = _DummyReq()
        req.kv.req_pool_idx = 0
        req.origin_input_ids = array("q", [1, 2, 3, 4, 5, 6])
        req.output_ids = array("q")
        req._kv_committed_len = len(req.origin_input_ids)
        kv_indices = allocator.alloc(req._kv_committed_len)
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, req._kv_committed_len)), kv_indices
        )
        req.extra_key = None
        req.cache_salt = None
        req.last_node = tree.root_node
        req.swa_uuid_for_lock = None
        req.kv.swa_evicted_seqlen = 0
        req.kv.cache_protected_len = 1
        # Intentionally mismatch to ensure code does not use len(prefix_indices).
        req.prefix_indices = torch.tensor([7, 8, 9, 10, 11], device=tree.device)

        captured = {}
        original_insert = tree.insert

        def wrapped_insert(params):
            captured["prev_prefix_len"] = params.prev_prefix_len
            captured["is_bigram"] = params.key.is_bigram
            captured["key_len"] = len(params.key)
            return original_insert(params)

        tree.insert = wrapped_insert
        tree.cache_finished_req(
            req, is_insert=True, kv_len_to_handle=req._kv_committed_len
        )

        self.assertEqual(captured["prev_prefix_len"], req.kv.cache_protected_len)
        self.assertTrue(captured["is_bigram"])
        self.assertEqual(captured["key_len"], len(req.origin_input_ids) - 1)

        # Case 2: is_insert=False should free [cache_protected_len:page_aligned_len]
        # even when len(prefix_indices) is intentionally larger.
        req2 = _DummyReq()
        req2.kv.req_pool_idx = 1
        req2.origin_input_ids = array("q", [11, 12, 13, 14, 15, 16])
        req2.output_ids = array("q")
        req2._kv_committed_len = len(req2.origin_input_ids)
        kv_indices2 = allocator.alloc(req2._kv_committed_len)
        req_to_token_pool.write(
            (req2.kv.req_pool_idx, slice(0, req2._kv_committed_len)), kv_indices2
        )
        req2.extra_key = None
        req2.cache_salt = None
        req2.last_node = tree.root_node
        req2.swa_uuid_for_lock = None
        req2.kv.swa_evicted_seqlen = 0
        req2.kv.cache_protected_len = 1
        req2.prefix_indices = torch.tensor([21, 22, 23, 24, 25], device=tree.device)

        freed_lens = []
        original_free = allocator.free

        def wrapped_free(indices):
            freed_lens.append(int(indices.numel()))
            return original_free(indices)

        allocator.free = wrapped_free
        tree.cache_finished_req(
            req2, is_insert=False, kv_len_to_handle=req2._kv_committed_len
        )

        # EAGLE + page_size=1 => page_aligned_len = committed_len - 1 = 5
        # Expected frees:
        #   overlap range [1:5] -> 4
        #   tail range [5:]     -> 1
        self.assertEqual(freed_lens, [4, 1])


# Optimization: SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.
# Splits a freshly-inserted leaf at the (page-aligned) sliding-window
# boundary so a future inc_lock_ref protects only ~sliding_window_size SWA
# tokens instead of the whole chunked-prefill chain.
class TestSWASplitLeafOnInsert(CustomTestCase):
    def _insert_and_lock(self, *, window, page_size, leaf_len, flag_on):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=128,
            kv_size_swa=64,
            sliding_window_size=window,
            page_size=page_size,
        )
        token_ids = list(range(leaf_len))
        with envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.override(flag_on):
            leaf = _insert_chain(tree, allocator, token_ids)
        result = tree.inc_lock_ref(leaf)
        return tree, leaf, result

    def test_flag_off_protects_full_leaf(self):
        tree, leaf, _ = self._insert_and_lock(
            window=4, page_size=1, leaf_len=12, flag_on=False
        )
        self.assertEqual(len(leaf.value), 12)
        self.assertEqual(tree.swa_protected_size_, 12)

    def test_flag_on_caps_protection_at_window(self):
        # (window, page_size, leaf_len, expected_tail_size); leaf_len picked
        # > tail_size and page-aligned for page_size > 1.
        cases = [
            (4, 1, 12, 4),
            (4, 1, 5, 4),
            (1, 1, 5, 1),
            (4, 2, 12, 4),
            (8, 2, 12, 8),
            (4, 4, 12, 4),
            # window NOT page-aligned -> tail rounds up to page boundary.
            (3, 2, 12, 4),
            (5, 4, 12, 8),
            (3, 4, 12, 4),
        ]
        for window, page_size, leaf_len, expected_tail in cases:
            with self.subTest(window=window, page_size=page_size, leaf_len=leaf_len):
                self.assertEqual(_expected_tail_size(window, page_size), expected_tail)
                tree, leaf, _ = self._insert_and_lock(
                    window=window,
                    page_size=page_size,
                    leaf_len=leaf_len,
                    flag_on=True,
                )
                self.assertEqual(len(leaf.value), expected_tail)
                self.assertEqual(tree.swa_protected_size_, expected_tail)

    def test_flag_on_no_split_when_leaf_within_window(self):
        # leaf_len <= tail_size: split must no-op.
        cases = [
            (4, 1, 4),
            (4, 1, 3),
            (4, 2, 4),
            (3, 2, 4),
            (8, 2, 4),
            (4, 4, 4),
        ]
        for window, page_size, leaf_len in cases:
            with self.subTest(window=window, page_size=page_size, leaf_len=leaf_len):
                tree, leaf, _ = self._insert_and_lock(
                    window=window,
                    page_size=page_size,
                    leaf_len=leaf_len,
                    flag_on=True,
                )
                self.assertEqual(len(leaf.value), leaf_len)
                self.assertEqual(tree.swa_protected_size_, leaf_len)

    def test_match_prefix_returns_full_chain_after_split(self):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=128,
            kv_size_swa=64,
            sliding_window_size=4,
            page_size=1,
        )
        token_ids = list(range(12))
        with envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.override(True):
            inserted_leaf = _insert_chain(tree, allocator, token_ids)
        self.assertEqual(len(inserted_leaf.value), 4)
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", token_ids)))
        )
        self.assertEqual(match.device_indices.shape[0], 12)
        self.assertIs(match.last_device_node, inserted_leaf)

    def test_dec_lock_ref_after_split_balances_to_zero(self):
        tree, leaf, result = self._insert_and_lock(
            window=4, page_size=1, leaf_len=12, flag_on=True
        )
        self.assertEqual(tree.swa_protected_size_, 4)
        self.assertEqual(tree.full_protected_size_, 12)

        tree.dec_lock_ref(
            leaf,
            params=DecLockRefParams(swa_uuid_for_lock=result.swa_uuid_for_lock),
        )

        self.assertEqual(tree.swa_protected_size_, 0)
        self.assertEqual(tree.full_protected_size_, 0)
        tree.sanity_check()


class _SinglePoolAllocator(BaseTokenToKVPoolAllocator):
    """Minimal single-pool allocator: no SWA peer, so the whole range dies
    together whatever the floor says."""

    def __init__(self):
        super().__init__(
            size=16,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index)


class TestFreeFullPartition(CustomTestCase):
    """`free_full` releases only the full side of a hybrid SWA allocator."""

    def setUp(self):
        _, self.allocator, _ = _build_swa_tree(is_eagle=False)
        self.full_baseline = self.allocator.full_available_size()
        self.swa_baseline = self.allocator.swa_available_size()

    def _sizes(self):
        return (
            self.allocator.full_available_size(),
            self.allocator.swa_available_size(),
        )

    def test_free_full_touches_only_the_full_pool(self):
        indices = _swa_alloc(self.allocator, 4)
        # free_full's precondition: the SWA peers are already released.
        self.allocator.free_swa(indices)
        self.assertEqual(self._sizes(), (self.full_baseline - 4, self.swa_baseline))

        self.allocator.free_full(indices)
        self.assertEqual(self._sizes(), (self.full_baseline, self.swa_baseline))

    def test_free_full_is_deferred_inside_a_free_group(self):
        indices = _swa_alloc(self.allocator, 4)
        self.allocator.free_swa(indices)

        self.allocator.free_group_begin()
        self.allocator.free_full(indices)
        self.assertEqual(self.allocator.full_available_size(), self.full_baseline - 4)
        self.allocator.free_group_end()

        self.assertEqual(self.allocator.full_available_size(), self.full_baseline)


class _RowCache:
    """Minimal PrefixCacheTrait host, so free_kv_row can be exercised without
    standing up a whole tree."""

    free_kv_row = BasePrefixCache.free_kv_row

    def __init__(self, allocator, row):
        self.req_to_token_pool = SimpleNamespace(req_to_token=row.unsqueeze(0))
        self.token_to_kv_pool_allocator = allocator
        self.page_size = allocator.page_size


class TestFreeKvRow(CustomTestCase):
    """A kv row is given back split at `swa_evicted_seqlen`: the full side
    whole, the SWA side only from the floor up."""

    def setUp(self):
        _, self.allocator, _ = _build_swa_tree(is_eagle=False)
        self.full_baseline = self.allocator.full_available_size()
        self.swa_baseline = self.allocator.swa_available_size()

    def _sizes(self):
        return (
            self.allocator.full_available_size(),
            self.allocator.swa_available_size(),
        )

    def test_floor_decides_how_much_of_the_swa_side_the_row_frees(self):
        # (start_pos, num_slots, floor, rows whose SWA peers are already gone)
        cases = [
            (0, 4, 4, 4),
            (8, 4, 8, 0),
            (8, 4, 10, 2),
            (8, 4, 4, 0),
        ]
        for start_pos, num_slots, floor, num_dead in cases:
            with self.subTest(start_pos=start_pos, floor=floor):
                indices = _swa_alloc(self.allocator, num_slots)
                # Window eviction already released the peers below the floor.
                if num_dead:
                    self.allocator.free_swa(indices[:num_dead])
                self.assertEqual(
                    self._sizes(),
                    (
                        self.full_baseline - num_slots,
                        self.swa_baseline - num_slots + num_dead,
                    ),
                )
                free_kv_row_segments(
                    self.allocator, [(indices, start_pos)], swa_evicted_seqlen=floor
                )
                self.assertEqual(self._sizes(), (self.full_baseline, self.swa_baseline))

    def test_adjacent_below_floor_pieces_release_their_shared_page_once(self):
        _, allocator, _ = _build_swa_tree(is_eagle=False, page_size=4)
        indices = _swa_alloc(allocator, 8)
        allocator.free_swa(indices)
        after_alloc = allocator.full_available_size()

        # Rows [0, 6) and [6, 8) both sit below the floor and share page 1.
        free_kv_row_segments(
            allocator,
            [(indices[:6], 0), (indices[6:], 6)],
            swa_evicted_seqlen=8,
        )

        self.assertEqual(allocator.full_available_size(), after_alloc + 8)

    def test_free_kv_row_reads_the_record_row_and_its_floor(self):
        indices = _swa_alloc(self.allocator, 8)
        cache = _RowCache(self.allocator, indices)
        kv = SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=3)
        self.allocator.free_swa(indices[:3])

        cache.free_kv_row(kv, [(1, 5)])

        # Rows [1, 5) go back on the full side; only [3, 5) still had SWA peers
        # to give back, so rows 5-7 keep the 3 SWA slots that are still out.
        self.assertEqual(self._sizes(), (self.full_baseline - 4, self.swa_baseline - 3))

    def test_single_pool_free_kv_row_still_frees_the_whole_range(self):
        allocator = _SinglePoolAllocator()
        cache = _RowCache(allocator, torch.arange(16, dtype=torch.int64))
        kv = SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=4)

        cache.free_kv_row(kv, [(2, 6)])

        self.assertEqual([t.tolist() for t in allocator.freed], [[2, 3], [4, 5]])

        # release_session and _free_kv_aligned dropped their own emptiness
        # guards, so an empty range has to stay a no-op here.
        cache.free_kv_row(kv, [(6, 6)])
        self.assertEqual(len(allocator.freed), 2)


class TestSWAPeerMappedContract(CustomTestCase):
    """page_size 1 gives back every peer the mapping names, without filtering:
    the contract replaces what `swa_indices > 0` used to absorb."""

    def _strict(self):
        return envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.STRICT))

    def _condition_checked_by(self, allocator, indices):
        """The predicate free_swa hands the async assert, as a python bool."""
        with self._strict():
            with mock.patch.object(torch, "_assert_async") as assert_async:
                allocator.free_swa(indices)
        return bool(assert_async.call_args.args[0])

    def test_free_swa_flags_a_slot_whose_peer_is_already_gone(self):
        _, allocator, _ = _build_swa_tree(is_eagle=False)
        live = _swa_alloc(allocator, 4)
        stale = _swa_alloc(allocator, 4)
        # Whoever released the peer left the mapping reading as the padding slot.
        allocator.clear_full_to_swa_mapping(stale)

        self.assertTrue(self._condition_checked_by(allocator, live))
        self.assertFalse(self._condition_checked_by(allocator, stale))

    @unittest.skipUnless(torch.cuda.is_available(), "sync detection needs CUDA")
    def test_free_swa_does_not_synchronize(self):
        """The filter's output shape was data-dependent, so it read a count back
        to the host; the gather that replaced it has a fixed shape."""
        _, allocator, _ = _build_swa_tree(is_eagle=False)
        mapping = allocator.full_to_swa_index_mapping

        # Warm up outside the window: a first-time cudaMalloc can synchronize on
        # its own, which the detector would report as this call's fault.
        allocator.free_swa(_swa_alloc(allocator, 4))
        indices = _swa_alloc(allocator, 4)

        # Gate on the pre-fix form: a detector blind to this sync class would pass
        # the assert below no matter how free_swa reads the mapping.
        peers = mapping[indices]
        if _sync_error(lambda: peers[peers > 0]) is None:
            self.skipTest("sync debug mode does not flag a data-dependent shape here")

        with self._strict():
            self.assertIsNone(_sync_error(lambda: allocator.free_swa(indices)))


class TestCacheUnfinishedReqEvictedPrefix(CustomTestCase):
    """An unfinished request whose SWA prefix is already gone must insert that
    prefix as a tombstone, not as live SWA KV."""

    def test_evicted_prefix_inserts_as_tombstone(self):
        page_size, window, num_tokens, evicted = 4, 4, 16, 8
        tree, allocator, req_to_token_pool = _build_swa_tree(
            is_eagle=False, page_size=page_size, sliding_window_size=window
        )
        kv_indices = _swa_alloc(allocator, num_tokens)
        req_to_token_pool.write((0, slice(0, num_tokens)), kv_indices)
        # Drop the prefix's SWA peers, as window eviction would.
        allocator.free_swa(kv_indices[:evicted])
        swa_before = allocator.swa_available_size()

        token_ids = array("q", range(1, num_tokens + 1))
        req = _DummyReq()
        req.kv.req_pool_idx = 0
        req.origin_input_ids = token_ids
        req.output_ids = array("q")
        req.get_fill_ids = lambda: token_ids
        req.extra_key = None
        req.cache_salt = None
        req.kv.cache_protected_len = 0
        req.last_node = tree.root_node
        req.swa_uuid_for_lock = None
        req.prefix_indices = torch.empty(0, dtype=torch.int64, device=tree.device)
        req.kv.swa_evicted_seqlen = evicted

        tree.cache_unfinished_req(req)

        # The insert itself frees nothing.
        self.assertEqual(allocator.swa_available_size(), swa_before)
        # The live leaf holds a full window, so the whole key stays matchable.
        self.assertEqual(req.kv.cache_protected_len, num_tokens)
        # [0, evicted) is a tombstone; only [evicted, num_tokens) counts as SWA.
        (first,) = tree.root_node.children.values()
        self.assertTrue(first.swa_tombstone)
        self.assertEqual(len(first.value), evicted)
        self.assertEqual(
            tree.swa_evictable_size_ + tree.swa_protected_size_,
            num_tokens - evicted,
        )

        # Finishing drops the locks, which sanity_check needs; the accounting
        # must survive the re-walk.
        tree.cache_finished_req(req, kv_len_to_handle=num_tokens)
        self.assertEqual(allocator.swa_available_size(), swa_before)
        self.assertEqual(
            tree.swa_evictable_size_ + tree.swa_protected_size_,
            num_tokens - evicted,
        )
        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
