"""Unit tests for SWA evict-prefix-keep-window strategy (unified path).

Tests the SGLANG_OPT_SWA_EVICT_KEEP_WINDOW path in
``SWAComponent.drive_eviction`` and the ``preserve_lru_position`` mode
of ``UnifiedRadixCache._split_node``:

* When the env var is off, eviction matches the legacy single-pass
  path (whole internal SWA chunks are tombstoned).
* When on, a long-SWA internal node is split at eviction time so the
  last ``ceil(sliding_window_size / page_size) * page_size`` tokens
  are retained as a smaller suffix child; only the older prefix half
  is tombstoned. "Retained" is identified by SIZE
  (``len(value) <= window_aligned``), no explicit marker — so any
  subsequent ``_split_node`` mid-walk produces two halves both
  ≤ window and both remain deferred without extra bookkeeping.
* A second pass evicts the window-sized nodes only if the
  first pass cannot satisfy the requested deficit.
* The split preserves the LRU position and ``last_access_time`` of
  the suffix child — splitting under memory pressure must not be
  treated as a fresh cache hit.

Setup notes:

* ``base`` uses 12 tokens (not 8) on purpose: the unified path
  unconditionally invokes ``SWAComponent._maybe_split_leaf_for_swa_lock``
  in ``commit_insert_component_data``, which splits any fresh leaf
  longer than ``tail_size = window_aligned``. With ``base = 12`` and
  ``window_aligned = 4`` the post-#26919 tree starts as
  ``root → A[1..8] (internal, value=8) → B[9..12] (leaf, value=4)``,
  which leaves an 8-token internal node ``A`` that our keep-window
  split can actually exercise. With ``base = 8`` ``A`` would already
  be 4 tokens (= window) and the split branch would never fire.

* ``match_prefix`` is intentionally avoided before ``evict`` — it
  would bump the long internal to MRU and let the chain leaf at LRU
  absorb the request. Long internals are fetched via
  ``tree.root_node.children`` instead.
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
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


def _build_unified_swa_tree(
    page_size: int = 2,
    sliding_window_size: int = 4,
    kv_size: int = 128,
    kv_size_swa: int = 128,
    max_context_len: int = 128,
):
    """Construct a UnifiedRadixCache with FULL + SWA components."""
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
    tree = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            sliding_window_size=sliding_window_size,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
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
    """Fetch the single child of `node` without touching LRU order."""
    assert len(node.children) == 1, f"expected one child, got {len(node.children)}"
    return next(iter(node.children.values()))


class TestSWAEvictKeepWindow(CustomTestCase):
    """Layout: page_size=2, sliding_window_size=4 → window_aligned=4.

    After #26919's leaf-cap split, inserting ``base = [1..12]`` yields
    ``root → A[1..8] (internal, value=8) → B[9..12] (leaf, value=4)``.
    A is the long internal our keep-window split targets.
    """

    def _swa_value_len(self, node):
        v = node.component_data[ComponentType.SWA].value
        return 0 if v is None else len(v)

    def test_keep_window_off_legacy_path(self):
        """Env off: internal SWA evicted wholesale, no keep-window split."""
        tree, allocator, _ = _build_unified_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(False):
            base = _make_seq(1, 12)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)

            base_node = _only_child(tree.root_node)
            self.assertEqual(self._swa_value_len(base_node), 8)
            self.assertGreater(len(base_node.children), 0)

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=8))

            # Whole SWA tombstoned, no structural split (key unchanged).
            self.assertIsNone(base_node.component_data[ComponentType.SWA].value)
            self.assertEqual(len(base_node.key), 8)
        tree.sanity_check()

    def test_keep_window_on_splits_and_retains_suffix(self):
        """Env on: long internal node split. Prefix tombstoned, last
        window-aligned tokens retained as a suffix child."""
        tree, allocator, _ = _build_unified_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 12)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)

            base_node_pre = _only_child(tree.root_node)
            self.assertEqual(self._swa_value_len(base_node_pre), 8)

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            # After split: root → new_parent ([1..4], SWA tombstoned)
            #              → suffix ([5..8], SWA alive, == base_node_pre).
            new_parent = _only_child(tree.root_node)
            self.assertEqual(len(new_parent.key), 4)
            self.assertIsNone(
                new_parent.component_data[ComponentType.SWA].value,
                "prefix half should be SWA-tombstoned",
            )
            suffix = _only_child(new_parent)
            self.assertIs(
                suffix,
                base_node_pre,
                "split keeps the original node as the suffix child",
            )
            self.assertEqual(len(suffix.key), 4)
            self.assertEqual(self._swa_value_len(suffix), 4)
        tree.sanity_check()

    def test_keep_window_eventually_drains_window_sized(self):
        """Under sustained pressure, pass 2 drains what pass 1 has
        deferred. Verifies that the retained suffix doesn't permanently
        wedge the SWA pool.

        Note: this asserts the END state (``swa_evictable_size == 0``),
        not specifically that pass-2 evicted the suffix AS an internal
        node — in this fixture pass 1 first deletes the chain leaf (a
        D-leaf) wholesale, leaving the suffix childless. The suffix
        remains deferred by size and pass 2 evicts it via the leaf-delete
        branch. Either way the suffix is gone, which is what the
        keep-window contract promises.
        """
        tree, allocator, _ = _build_unified_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 12)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)

            # Pass 1 splits base_node and tombstones the prefix (4 tokens).
            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            # Now request all remaining SWA. Pass 1 alone can't satisfy
            # this (all surviving internals are window-sized → deferred).
            # Pass 2 picks them up.
            remaining = tree.swa_evictable_size()
            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=remaining))
            self.assertEqual(tree.swa_evictable_size(), 0)
        tree.sanity_check()

    def test_window_sized_leaf_deferred_in_pass1(self):
        """Pass 1 must defer any SWA node with ``len(value) ≤
        window_aligned``, including D-leaves. This is the structural
        guarantee that protects a retained suffix even after its
        descendants are evicted and it becomes a leaf itself
        (otherwise the next pass-1 scan would atom-delete exactly the
        window we wanted to keep).

        Direct test of the invariant: insert a window-sized leaf that
        sits at the LRU end, plus a longer branch pass 1 can pull from.
        Pass 1 must skip the short leaf and evict from the long branch.
        """
        tree, allocator, _ = _build_unified_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            # Short standalone leaf (== window_aligned tokens; #26919's
            # leaf-cap split returns early since len ≤ tail_size).
            short = _make_seq(1, 4)
            _insert(tree, allocator, short)
            short_node = _only_child(tree.root_node)
            self.assertEqual(self._swa_value_len(short_node), 4)
            self.assertIn(short_node, tree.evictable_device_leaves)

            # Long branch: #26919 splits the 12-token leaf into an
            # 8-token internal + 4-token leaf, then the chain extension
            # creates more internals. Pass 1 has a long internal it can
            # split-and-tombstone instead of touching short_node.
            long_base = _make_seq(2000, 12)
            long_chain = long_base + _make_seq(3000, 8)
            _insert(tree, allocator, long_base)
            _insert(tree, allocator, long_chain)
            self.assertIs(
                tree.lru_lists[ComponentType.SWA].get_lru_no_lock(),
                short_node,
                "short window-sized leaf must be the pass-1 candidate",
            )

            # Small request — pass 1 alone must satisfy by splitting the
            # long internal (frees 4 SWA), without atom-deleting
            # short_node.
            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            self.assertIsNotNone(
                short_node.component_data[ComponentType.SWA].value,
                "window-sized leaf must be deferred in pass 1",
            )
            self.assertEqual(self._swa_value_len(short_node), 4)
            self.assertIn(short_node, tree.evictable_device_leaves)
        tree.sanity_check()

    def test_split_preserves_lru_position_and_access_time(self):
        """``preserve_lru_position=True``: the suffix child stays at the
        original LRU slot and keeps its ``last_access_time``. Splitting
        under memory pressure must not be treated as a fresh cache hit
        (which insert_mru would imply)."""
        tree, allocator, _ = _build_unified_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 12)
            full_chain = base + _make_seq(900, 8)
            other_base = _make_seq(2000, 12)
            other_chain = other_base + _make_seq(3000, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)
            _insert(tree, allocator, other_base)
            _insert(tree, allocator, other_chain)

            swa_lru = tree.lru_lists[ComponentType.SWA]
            pt = swa_lru._pt
            # The 8-token internal A from the base branch is at LRU
            # (inserted first; nothing bumped it). Capture its raw next
            # pointer (toward tail) and access time so we can verify
            # they survive the split.
            old_lru = swa_lru.get_lru_no_lock()
            old_next = old_lru.lru_next[pt]
            old_access_time = old_lru.last_access_time
            self.assertEqual(self._swa_value_len(old_lru), 8)

            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            # After split: suffix sits at the original LRU position.
            suffix = swa_lru.get_lru_no_lock()
            self.assertIs(
                suffix,
                old_lru,
                "suffix child should reuse the original LRU slot",
            )
            self.assertIs(
                suffix.lru_next[pt],
                old_next,
                "suffix's LRU-side neighbor must be unchanged",
            )
            self.assertEqual(
                suffix.last_access_time,
                old_access_time,
                "last_access_time must be preserved (no MRU promotion)",
            )
            self.assertIsNotNone(suffix.component_data[ComponentType.SWA].value)
        tree.sanity_check()

    def test_mid_split_keeps_both_halves_under_window(self):
        """Regression for the marker-propagation worry: with size-based
        identification, if a retained suffix is later split mid-node
        (e.g. by an overlapping insert), BOTH halves are still
        ≤ window_aligned by construction, so pass 1 keeps deferring
        both — no marker propagation needed.
        """
        tree, allocator, _ = _build_unified_swa_tree()
        with envs.SGLANG_OPT_SWA_EVICT_KEEP_WINDOW.override(True):
            base = _make_seq(1, 12)
            full_chain = base + _make_seq(900, 8)
            _insert(tree, allocator, base)
            _insert(tree, allocator, full_chain)
            tree.evict(EvictParams(num_tokens=0, swa_num_tokens=4))

            # Tree now: root → tombstoned_prefix ([1..4]) → retained
            # ([5..8]) → ... (chain below).
            tombstoned_prefix = _only_child(tree.root_node)
            retained = _only_child(tombstoned_prefix)
            self.assertEqual(len(retained.key), 4)
            self.assertIsNotNone(
                retained.component_data[ComponentType.SWA].value
            )

            # Insert sibling = base[:6] + [700, 701]: walks through
            # tombstoned_prefix (full match, gets revived), then matches
            # 2/4 of retained's key → _split_node fires inside retained
            # at position 2, yielding split_parent ([5,6]) + retained' ([7,8]).
            sibling = base[:6] + _make_seq(700, 2)
            _insert(tree, allocator, sibling)

            self.assertEqual(len(retained.key), 2, "retained becomes 2-token suffix")
            split_parent = retained.parent
            self.assertEqual(len(split_parent.key), 2)
            self.assertIs(split_parent.parent, tombstoned_prefix)

            # Both halves are ≤ window_aligned = 4 → pass 1 defers both.
            window_aligned = 4
            self.assertLessEqual(self._swa_value_len(split_parent), window_aligned)
            self.assertLessEqual(self._swa_value_len(retained), window_aligned)
            # Both retain alive SWA values (deferral didn't lose them).
            self.assertIsNotNone(
                split_parent.component_data[ComponentType.SWA].value
            )
            self.assertIsNotNone(
                retained.component_data[ComponentType.SWA].value
            )
        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
