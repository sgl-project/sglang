"""Unit tests for legacy SWARadixCache evict-prefix-keep-window strategy.

Mirrors test_swa_evict_keep_window.py (unified path) but targets the
legacy SWARadixCache. Verifies:

* Env off: legacy single-pass eviction preserved verbatim.
* Env on: long internal node split, suffix retained at the original
  LRU slot. "Retained" is identified by SIZE
  (``len(value) <= window_aligned``), no explicit marker.
* Pass 2 drains window-sized nodes when pass 1 alone can't
  satisfy the deficit.
* ``_split_node(preserve_lru_position=True)`` keeps the suffix at the
  original LRU position and last_access_time (no MRU promotion).
* A subsequent mid-node split (e.g. via overlapping insert) leaves
  BOTH halves ≤ window — the structural invariant carries the
  retention semantics without a marker.

Setup avoids ``match_prefix`` before ``evict``: the legacy SWA
eviction freely deletes leaves, so a match_prefix-bumped LRU order
(internal at MRU, chain leaf at LRU) would let the leaf absorb the
request and never split the internal node we want to test.
"""

import unittest
from array import array

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


def _build_swa_tree(
    page_size: int = 2,
    sliding_window_size: int = 4,
    kv_size: int = 128,
    kv_size_swa: int = 128,
    max_context_len: int = 128,
):
    head_num = 2
    head_dim = 64
    num_layers = 8
    dtype = torch.bfloat16
    device = get_device()
    full_attention_layer_ids = list(range(0, num_layers, 2))
    swa_attention_layer_ids = [
        i for i in range(num_layers) if i not in set(full_attention_layer_ids)
    ]

    req_to_token_pool = ReqToTokenPool(
        size=8,
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
        enable_kvcache_transpose=False,
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
            is_eagle=False,
            sliding_window_size=sliding_window_size,
        ),
    )
    return tree, allocator, req_to_token_pool


def _alloc(allocator, need_size: int):
    assert need_size % allocator.page_size == 0
    full_indices = allocator.full_attn_allocator.alloc(need_size)
    swa_indices = allocator.swa_attn_allocator.alloc(need_size)
    assert full_indices is not None and swa_indices is not None
    allocator.full_to_swa_index_mapping[full_indices] = swa_indices
    return full_indices


def _insert(tree, allocator, token_ids):
    indices = _alloc(allocator, len(token_ids))
    tree.insert(InsertParams(key=RadixKey(array("q", token_ids)), value=indices))


def _make_seq(start: int, length: int) -> list[int]:
    return list(range(start, start + length))


def _only_child(node):
    assert len(node.children) == 1, f"expected one child, got {len(node.children)}"
    return next(iter(node.children.values()))


class TestSWARadixKeepWindow(CustomTestCase):
    """Layout: page_size=2, sliding_window_size=4 → window_aligned=4.

    Chain: root → A (8 tokens) → B (8 tokens). A is internal with
    ``len(value) = 8 > 4``, so pass 1 splits it under keep-window.
    """

    def test_keep_window_off_legacy_path(self):
        """Env off: internal SWA evicted wholesale, no split."""
        tree, allocator, _ = _build_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(False):
            base = _make_seq(1, 8)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)

            base_node = _only_child(tree.root_node)
            self.assertGreater(len(base_node.children), 0)
            self.assertEqual(len(base_node.value), 8)
            self.assertFalse(base_node.swa_tombstone)

            # base_node at LRU (chain_leaf is MRU), so the internal-
            # tombstone path fires after just 8 tokens — the leaf isn't
            # touched.
            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=8))

            self.assertTrue(base_node.swa_tombstone)
            self.assertEqual(len(base_node.key), 8)
        tree.sanity_check()

    def test_keep_window_on_splits_and_retains_suffix(self):
        """Env on: long internal node split. Prefix tombstoned, last
        window-aligned tokens retained as a suffix child."""
        tree, allocator, _ = _build_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 8)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)

            base_node_pre = _only_child(tree.root_node)
            self.assertEqual(len(base_node_pre.value), 8)

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            # After split: root → new_parent ([1..4], tombstoned)
            #              → suffix ([5..8], alive, == base_node_pre).
            new_parent = _only_child(tree.root_node)
            self.assertEqual(len(new_parent.key), 4)
            self.assertTrue(
                new_parent.swa_tombstone, "prefix half should be SWA-tombstoned"
            )
            suffix = _only_child(new_parent)
            self.assertIs(suffix, base_node_pre)
            self.assertEqual(len(suffix.key), 4)
            self.assertEqual(len(suffix.value), 4)
            self.assertFalse(suffix.swa_tombstone)
        tree.sanity_check()

    def test_keep_window_eventually_drains_window_sized(self):
        """Under sustained pressure, pass 2 drains what pass 1 has
        deferred. Verifies that the retained suffix doesn't permanently
        wedge the SWA pool.

        Note: this asserts the END state (``swa_evictable_size == 0``),
        not specifically that pass-2 evicted the suffix AS an internal
        node — in this fixture pass 1 first deletes ``chain_leaf`` (a
        leaf) wholesale, leaving the suffix childless. The suffix remains
        deferred by size and pass 2 evicts it via the leaf-delete branch.
        Either way the suffix is gone, which is what the keep-window
        contract promises.
        """
        tree, allocator, _ = _build_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 8)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))
            remaining = tree.swa_evictable_size()
            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=remaining))
            self.assertEqual(tree.swa_evictable_size(), 0)
        tree.sanity_check()

    def test_window_sized_suffix_leaf_deferred_in_pass1(self):
        """A retained suffix can become a leaf after its children are
        evicted. It must keep first-pass protection; otherwise a later
        SWA eviction deletes exactly the window we retained.
        """
        tree, allocator, _ = _build_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 8)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)
            # Capture the base-chain top BEFORE adding the other branch:
            # afterwards root will have two children and ``_only_child``
            # no longer works. The captured Python object is the same one
            # that ``_split_node`` will later rename in-place as the
            # retained suffix.
            base_chain_top = _only_child(tree.root_node)
            other = _make_seq(2000, 8)
            _insert(tree, allocator, other)

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))
            # Same Python object, now sliced to the window-sized suffix.
            suffix = base_chain_top
            tombstoned_prefix = suffix.parent
            self.assertTrue(tombstoned_prefix.swa_tombstone)
            self.assertEqual(len(suffix.key), 4)
            self.assertGreater(len(suffix.children), 0)

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=8))
            self.assertEqual(len(suffix.children), 0)
            self.assertFalse(suffix.swa_tombstone)
            self.assertIn(suffix.id, tree.swa_lru_list.cache)
            self.assertIs(
                tree.swa_lru_list.get_lru_no_lock(),
                suffix,
                "retained suffix leaf must be the pass-1 candidate",
            )

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))
            self.assertFalse(suffix.swa_tombstone)
            self.assertIn(suffix.id, tree.swa_lru_list.cache)
        tree.sanity_check()

    def test_split_preserves_lru_position_and_access_time(self):
        """``preserve_lru_position=True``: suffix child keeps the
        original LRU slot and ``last_access_time``."""
        tree, allocator, _ = _build_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 8)
            full_chain = base + _make_seq(900, 8)
            other_base = _make_seq(2000, 8)
            other_chain = other_base + _make_seq(3000, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)
            _insert(tree, allocator, other_base)
            _insert(tree, allocator, other_chain)

            swa_lru = tree.swa_lru_list
            old_lru = swa_lru.get_lru_no_lock()
            old_next = old_lru.swa_next
            old_access_time = old_lru.last_access_time

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            suffix = swa_lru.get_lru_no_lock()
            self.assertIs(
                suffix, old_lru, "suffix child should reuse the original LRU slot"
            )
            self.assertIs(
                suffix.swa_next,
                old_next,
                "suffix's LRU-side neighbor must be unchanged",
            )
            self.assertEqual(
                suffix.last_access_time,
                old_access_time,
                "last_access_time must be preserved (no MRU promotion)",
            )
            self.assertFalse(suffix.swa_tombstone)
        tree.sanity_check()

    def test_mid_split_keeps_both_halves_under_window(self):
        """With size-based identification, a subsequent ``_split_node``
        on the retained suffix yields two halves both ≤ window, so
        pass 1 keeps deferring both. No marker needed.
        """
        tree, allocator, _ = _build_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 8)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)
            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            tombstoned_prefix = _only_child(tree.root_node)
            retained = _only_child(tombstoned_prefix)
            self.assertEqual(len(retained.key), 4)
            self.assertFalse(retained.swa_tombstone)

            # Insert sibling triggers mid-split of the retained suffix.
            sibling = base[:6] + _make_seq(700, 2)
            _insert(tree, allocator, sibling)

            self.assertEqual(len(retained.key), 2)
            split_parent = retained.parent
            self.assertEqual(len(split_parent.key), 2)
            self.assertIs(split_parent.parent, tombstoned_prefix)

            window_aligned = 4
            self.assertLessEqual(len(split_parent.value), window_aligned)
            self.assertLessEqual(len(retained.value), window_aligned)
            self.assertFalse(split_parent.swa_tombstone)
            self.assertFalse(retained.swa_tombstone)
        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
