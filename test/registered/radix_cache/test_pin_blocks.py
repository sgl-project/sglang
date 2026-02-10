"""
Unit tests for PIN/UNPIN block operations on RadixCache.

Tests the block_hash_index, pin_blocks, and unpin_blocks methods.
Uses RadixCache.create_simulated() -- no GPU or server required.

How PIN works:
  pin_blocks(hash) -> inc_lock_ref(node) -> node removed from evictable_leaves
  -> evict() skips it because it's not in evictable_leaves
  -> unpin_blocks(hash) -> dec_lock_ref(node) -> node re-added to evictable_leaves

Without load or explicit eviction, nodes stay in cache forever. These tests
force eviction via evict(num_tokens=BIG_NUMBER) and verify that:
  - Pinned nodes survive (lock_ref > 0, not in evictable_leaves)
  - Unpinned nodes get evicted (value set to None)

Usage:
    python -m pytest test/registered/radix_cache/test_pin_blocks.py -v
"""

import unittest
import unittest.mock

import torch

from sglang.srt.disaggregation.kv_events import BlockStored
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


def _get_block_hashes(cache: RadixCache) -> list[int]:
    """Extract all block hashes from BlockStored events."""
    events = cache.take_events()
    hashes = []
    for e in events:
        if isinstance(e, BlockStored):
            hashes.extend(e.block_hashes)
    return hashes


def _dump_tree(cache: RadixCache, label: str = "") -> str:
    """Dump the radix tree state for debugging.

    Returns a human-readable string showing:
      - Tree structure with indentation
      - Node key (token IDs), lock_ref, evicted status
      - Whether each node is in evictable_leaves
      - block_hash_index contents
      - external_pin_count contents

    Example output:
      === TREE STATE: after pin ===
        root (lock_ref=1)
          [1,2,3,4] lock_ref=1 evicted=False EVICTABLE=No
            [5,6,7,8] lock_ref=1 evicted=False EVICTABLE=No  <-- PINNED
          [9,10,11,12] lock_ref=0 evicted=False EVICTABLE=Yes
      block_hash_index: {hash1: node_id_2, hash2: node_id_3, hash3: node_id_4}
      external_pin_count: {hash2: 1}
      evictable_leaves: {node_id_4}
    """
    lines = []
    if label:
        lines.append(f"=== TREE STATE: {label} ===")
    else:
        lines.append("=== TREE STATE ===")

    # Walk tree
    stack = [(cache.root_node, 2)]
    while stack:
        node, indent = stack.pop()
        is_root = node is cache.root_node
        in_evictable = node in cache.evictable_leaves
        pinned_hashes = set()
        if node.hash_value:
            from sglang.srt.mem_cache.radix_cache import hash_str_to_int64
            for h in node.hash_value:
                int_hash = hash_str_to_int64(h)
                if int_hash in cache.external_pin_count:
                    pinned_hashes.add(int_hash)

        if is_root:
            lines.append(f"  root (lock_ref={node.lock_ref})")
        else:
            tokens = list(node.key.token_ids[:16])
            pin_marker = "  <-- PINNED" if pinned_hashes else ""
            lines.append(
                f"{' ' * indent}[{','.join(str(t) for t in tokens)}] "
                f"lock_ref={node.lock_ref} evicted={node.evicted} "
                f"EVICTABLE={'Yes' if in_evictable else 'No'}"
                f"{pin_marker}"
            )

        for child in node.children.values():
            stack.append((child, indent + 4))

    # Summaries
    lines.append(
        f"  block_hash_index: {len(cache.block_hash_index)} entries"
    )
    lines.append(
        f"  external_pin_count: {dict(cache.external_pin_count)}"
    )
    evictable_ids = [n.id for n in cache.evictable_leaves]
    lines.append(
        f"  evictable_leaves: {len(cache.evictable_leaves)} nodes (ids={evictable_ids})"
    )

    result = "\n".join(lines)
    print(result)  # Also print for pytest -s visibility
    return result


def _collect_nodes(cache: RadixCache) -> list[TreeNode]:
    """Collect all non-root nodes in the tree."""
    nodes = []
    stack = list(cache.root_node.children.values())
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in node.children.values():
            stack.append(child)
    return nodes


class TestPinBlocks(unittest.TestCase):
    """Test cases for pin_blocks / unpin_blocks."""

    def _make_cache(self, page_size: int = 4) -> RadixCache:
        mock_allocator = unittest.mock.Mock()
        mock_allocator.free = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")
        return RadixCache.create_simulated(
            page_size=page_size,
            enable_kv_cache_events=True,
            mock_allocator=mock_allocator,
        )

    # ---- Index tests ----

    def test_block_hash_index_populated_on_insert(self):
        """Verify block_hash_index is populated when nodes are inserted."""
        cache = self._make_cache(page_size=4)
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4]), value=None))

        hashes = _get_block_hashes(cache)
        self.assertEqual(len(hashes), 1)  # 4 tokens / page_size 4 = 1 block
        self.assertIn(hashes[0], cache.block_hash_index)

    def test_block_hash_index_multiple_pages(self):
        """Verify index has entries for multi-page inserts."""
        cache = self._make_cache(page_size=2)
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4, 5, 6]), value=None))

        hashes = _get_block_hashes(cache)
        self.assertEqual(len(hashes), 3)  # 6 tokens / page_size 2 = 3 blocks
        for h in hashes:
            self.assertIn(h, cache.block_hash_index)

    def test_index_cleared_on_reset(self):
        """Reset clears block_hash_index and external_pin_count."""
        cache = self._make_cache(page_size=4)
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4]), value=None))
        hashes = _get_block_hashes(cache)
        cache.pin_blocks(hashes)

        self.assertTrue(len(cache.block_hash_index) > 0)
        self.assertTrue(len(cache.external_pin_count) > 0)

        cache.reset()

        self.assertEqual(len(cache.block_hash_index), 0)
        self.assertEqual(len(cache.external_pin_count), 0)

    def test_index_cleared_on_evict(self):
        """Evicting a node removes its hashes from block_hash_index."""
        cache = self._make_cache(page_size=4)
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4]), value=None))
        hashes = _get_block_hashes(cache)

        self.assertIn(hashes[0], cache.block_hash_index)
        _dump_tree(cache, "before eviction")

        cache.evict(EvictParams(num_tokens=100))
        _dump_tree(cache, "after eviction")

        self.assertNotIn(hashes[0], cache.block_hash_index)

    # ---- PIN / UNPIN edge cases ----

    def test_pin_unknown_hash(self):
        """Pinning a hash that doesn't exist returns 0."""
        cache = self._make_cache()
        pinned = cache.pin_blocks([999999])
        self.assertEqual(pinned, 0)

    def test_unpin_unknown_hash(self):
        """Unpinning a hash that was never pinned returns 0."""
        cache = self._make_cache()
        unpinned = cache.unpin_blocks([999999])
        self.assertEqual(unpinned, 0)

    # ---- Core PIN behavior: internal state verification ----

    def test_pin_sets_lock_ref_and_removes_from_evictable(self):
        """PIN must: set lock_ref > 0 AND remove node from evictable_leaves.

        This is the fundamental mechanism -- without this, evict() would
        still pick up the node regardless of any 'pinned' flag.
        """
        cache = self._make_cache(page_size=4)
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4]), value=None))
        hashes = _get_block_hashes(cache)

        # Before pin: node should be evictable (lock_ref=0, in evictable_leaves)
        node = cache.block_hash_index[hashes[0]]
        self.assertEqual(node.lock_ref, 0, "Before pin: lock_ref should be 0")
        self.assertIn(node, cache.evictable_leaves, "Before pin: node should be evictable")
        _dump_tree(cache, "before pin")

        # Pin it
        pinned = cache.pin_blocks(hashes)
        self.assertEqual(pinned, 1)

        # After pin: lock_ref > 0, NOT in evictable_leaves
        self.assertGreater(node.lock_ref, 0, "After pin: lock_ref must be > 0")
        self.assertNotIn(node, cache.evictable_leaves, "After pin: node must NOT be evictable")
        self.assertEqual(cache.external_pin_count[hashes[0]], 1)
        _dump_tree(cache, "after pin")

    def test_unpin_restores_evictable_status(self):
        """UNPIN must: restore lock_ref to 0 AND re-add node to evictable_leaves."""
        cache = self._make_cache(page_size=4)
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4]), value=None))
        hashes = _get_block_hashes(cache)

        node = cache.block_hash_index[hashes[0]]

        cache.pin_blocks(hashes)
        self.assertNotIn(node, cache.evictable_leaves)

        cache.unpin_blocks(hashes)
        self.assertEqual(node.lock_ref, 0, "After unpin: lock_ref should be 0")
        self.assertIn(node, cache.evictable_leaves, "After unpin: node should be evictable again")
        self.assertNotIn(hashes[0], cache.external_pin_count, "After unpin: external_pin_count should be cleared")
        _dump_tree(cache, "after unpin")

    # ---- Core PIN behavior: eviction resistance ----

    def test_pin_survives_eviction_unpin_does_not(self):
        """The definitive test: insert two sequences, pin one, evict everything.

        Pinned sequence A stays in tree (reachable from root, in block_hash_index).
        Unpinned sequence B is removed from tree (not reachable, gone from index).

        This proves PIN actually matters -- without it, both would be evicted.

        Note: evict() calls _delete_leaf() which removes the node from
        parent.children but does NOT set node.value = None. So we check tree
        reachability (via _collect_nodes and match_prefix), not node.evicted.
        """
        cache = self._make_cache(page_size=4)

        # Insert sequence A: [1,2,3,4]
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4]), value=None))
        hashes_a = _get_block_hashes(cache)

        # Insert sequence B: [5,6,7,8]
        cache.insert(InsertParams(key=RadixKey([5, 6, 7, 8]), value=None))
        hashes_b = _get_block_hashes(cache)

        node_a = cache.block_hash_index[hashes_a[0]]
        node_b = cache.block_hash_index[hashes_b[0]]

        _dump_tree(cache, "after insert both -- both evictable")

        # Both nodes should be in the tree
        tree_nodes = _collect_nodes(cache)
        self.assertIn(node_a, tree_nodes, "A should be in tree")
        self.assertIn(node_b, tree_nodes, "B should be in tree")

        # Pin A only
        cache.pin_blocks(hashes_a)
        _dump_tree(cache, "after pinning A -- B still evictable")

        # Verify pre-eviction state
        self.assertNotIn(node_a, cache.evictable_leaves, "A should NOT be evictable (pinned)")
        self.assertIn(node_b, cache.evictable_leaves, "B SHOULD be evictable (not pinned)")

        # Force eviction of everything possible
        cache.evict(EvictParams(num_tokens=999))
        _dump_tree(cache, "after evict(999) -- A survives, B gone")

        # A survives: still in tree, still in block_hash_index
        tree_nodes = _collect_nodes(cache)
        self.assertIn(node_a, tree_nodes, "Pinned node A must still be in tree")
        self.assertIn(hashes_a[0], cache.block_hash_index, "Pinned hash A must still be in index")

        # B is gone: removed from tree and block_hash_index
        self.assertNotIn(node_b, tree_nodes, "Unpinned node B must be removed from tree")
        self.assertNotIn(hashes_b[0], cache.block_hash_index, "Evicted hash B must be removed from index")

        # Verify via match_prefix too
        result_a = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        self.assertGreater(len(result_a.device_indices), 0, "Pinned A should still match")

        result_b = cache.match_prefix(MatchPrefixParams(key=RadixKey([5, 6, 7, 8])))
        self.assertEqual(len(result_b.device_indices), 0, "Evicted B should not match")

        # Now unpin A and evict again -- A should be gone too
        cache.unpin_blocks(hashes_a)
        _dump_tree(cache, "after unpin A -- now evictable")
        self.assertIn(node_a, cache.evictable_leaves, "A should be evictable after unpin")

        cache.evict(EvictParams(num_tokens=999))
        _dump_tree(cache, "after second evict -- everything gone")

        tree_nodes = _collect_nodes(cache)
        self.assertNotIn(node_a, tree_nodes, "A must be removed from tree after unpin + evict")
        self.assertEqual(len(tree_nodes), 0, "Tree should be empty after full eviction")

    # ---- Double pin (refcount) ----

    def test_double_pin_requires_double_unpin(self):
        """Pinning twice increments refcount. Must unpin twice to make evictable."""
        cache = self._make_cache(page_size=4)
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4]), value=None))
        hashes = _get_block_hashes(cache)
        node = cache.block_hash_index[hashes[0]]

        # Pin twice
        cache.pin_blocks(hashes)
        cache.pin_blocks(hashes)
        self.assertEqual(cache.external_pin_count[hashes[0]], 2)
        self.assertGreater(node.lock_ref, 0)
        _dump_tree(cache, "pinned x2")

        # First unpin: count drops to 1, still pinned
        cache.unpin_blocks(hashes)
        self.assertEqual(cache.external_pin_count[hashes[0]], 1)
        self.assertGreater(node.lock_ref, 0, "Still pinned with count=1")
        self.assertNotIn(node, cache.evictable_leaves, "Still not evictable")
        _dump_tree(cache, "after first unpin (count=1)")

        # Attempt eviction -- should fail to evict the pinned node
        cache.evict(EvictParams(num_tokens=999))
        tree_nodes = _collect_nodes(cache)
        self.assertIn(node, tree_nodes, "Node must survive eviction while pin count > 0")
        _dump_tree(cache, "after evict attempt -- still pinned")

        # Second unpin: fully unpinned
        cache.unpin_blocks(hashes)
        self.assertNotIn(hashes[0], cache.external_pin_count)
        self.assertEqual(node.lock_ref, 0)
        self.assertIn(node, cache.evictable_leaves)
        _dump_tree(cache, "after second unpin -- now evictable")

        # Now eviction works
        cache.evict(EvictParams(num_tokens=999))
        tree_nodes = _collect_nodes(cache)
        self.assertNotIn(node, tree_nodes, "Fully unpinned node must be evicted from tree")
        _dump_tree(cache, "after evict -- gone")

    # ---- Ancestor protection ----

    def test_pin_protects_ancestors(self):
        """Pinning a leaf protects its ancestor chain via lock_ref propagation.

        Tree structure after inserts:
          root
            [1,2,3,4]          <-- shared prefix (parent of both branches)
              [5,6,7,8]        <-- branch A (we pin this)
              [9,10,11,12]     <-- branch B (should be evictable)

        Pinning [5,6,7,8] increments lock_ref on [5,6,7,8] AND its parent
        [1,2,3,4]. So [1,2,3,4] is also protected. Only [9,10,11,12] can
        be evicted.
        """
        cache = self._make_cache(page_size=4)

        # Insert first branch: [1,2,3,4,5,6,7,8]
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4, 5, 6, 7, 8]), value=None))
        hashes_first = _get_block_hashes(cache)  # 2 blocks
        self.assertEqual(len(hashes_first), 2, "Should get 2 block hashes for 8 tokens / page_size 4")

        # Insert second branch: [1,2,3,4,9,10,11,12] -- shares [1,2,3,4] prefix
        cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4, 9, 10, 11, 12]), value=None))
        hashes_second = _get_block_hashes(cache)  # 1 new block (prefix already exists)

        _dump_tree(cache, "branching tree -- both branches evictable")

        # Identify nodes
        node_prefix = cache.block_hash_index[hashes_first[0]]   # [1,2,3,4]
        node_branch_a = cache.block_hash_index[hashes_first[1]]  # [5,6,7,8]
        node_branch_b = cache.block_hash_index[hashes_second[0]] # [9,10,11,12]

        # Pin branch A leaf [5,6,7,8]
        cache.pin_blocks([hashes_first[1]])
        _dump_tree(cache, "after pinning branch A [5,6,7,8]")

        # Verify lock_ref propagation: both node_branch_a AND node_prefix have lock_ref > 0
        self.assertGreater(node_branch_a.lock_ref, 0, "Pinned node must have lock_ref > 0")
        self.assertGreater(node_prefix.lock_ref, 0, "Parent of pinned node must have lock_ref > 0 (ancestor protection)")
        self.assertEqual(node_branch_b.lock_ref, 0, "Unpinned branch should have lock_ref = 0")

        # Branch B should be evictable, branch A and prefix should not
        self.assertIn(node_branch_b, cache.evictable_leaves, "Unpinned branch B should be evictable")
        self.assertNotIn(node_branch_a, cache.evictable_leaves, "Pinned branch A should not be evictable")
        # Prefix is not a leaf (has children), so it's never in evictable_leaves regardless
        self.assertNotIn(node_prefix, cache.evictable_leaves, "Prefix node is not a leaf")

        # Evict everything possible
        cache.evict(EvictParams(num_tokens=999))
        _dump_tree(cache, "after evict -- branch B gone, A + prefix survive")

        # Branch B evicted (removed from tree)
        tree_nodes = _collect_nodes(cache)
        self.assertNotIn(node_branch_b, tree_nodes, "Unpinned branch B must be removed from tree")
        # Branch A + prefix survive (still in tree)
        self.assertIn(node_branch_a, tree_nodes, "Pinned branch A must survive")
        self.assertIn(node_prefix, tree_nodes, "Protected prefix must survive")


if __name__ == "__main__":
    unittest.main()
