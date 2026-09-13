"""CPU-only unit tests for coverage-preserving Mamba device eviction."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from array import array
from collections import defaultdict

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MAMBA_REUSED_KEY,
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
    LRURefreshPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList, UnifiedTreeNode
from sglang.test.test_utils import CustomTestCase

MAMBA = ComponentType.MAMBA


class _FakeTreeCore:
    tree_components = (ComponentType.FULL, MAMBA)
    enable_session_radix_cache = False

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_device_leaves = set()
        self.component_evictable_size_ = {MAMBA: 0}
        self.component_protected_size_ = {MAMBA: 0}
        self.lru_lists = {MAMBA: UnifiedLRUList(MAMBA, self.tree_components)}
        self.host_lru_lists = {
            MAMBA: UnifiedLRUList(MAMBA, self.tree_components, use_host_ptr=True)
        }
        self.cascaded = []
        self._is_tracking_unbacked_tokens = False
        self._tracked_unbacked_tokens = 0

    def _evict_component_and_detach_lru(self, node, component, *args, **kwargs):
        return UnifiedTreeCore._evict_component_and_detach_lru(
            self, node, component, *args, **kwargs
        )

    def _cascade_evict(self, node, component, tracker, device_frees, host_frees):
        self.cascaded.append(node)


def _build_chain(key_lens):
    """A root-to-leaf chain, one mamba state per node, inserted into the LRU
    in chain order so the shallowest node is the LRU tail. State slot i is the
    node's 1-based chain position."""
    core = _FakeTreeCore()
    component = object.__new__(MambaComponent)
    component.tree_core = core
    nodes = []
    parent = core.root_node
    for index, key_len in enumerate(key_lens):
        node = UnifiedTreeNode(core.tree_components)
        node.parent = parent
        node.key = RadixKey(token_ids=array("q", [index] * key_len))
        node.component_data[ComponentType.FULL].value = torch.tensor([index])
        node.component_data[MAMBA].value = torch.tensor([index + 1])
        parent.children[index] = node
        core.component_evictable_size_[MAMBA] += 1
        core.lru_lists[MAMBA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, core


def _evict_one(component):
    """One eviction round of one slot; returns (leaf id or None, freed slots)."""
    device_frees = defaultdict(list)
    host_frees = defaultdict(list)
    tracker = {ComponentType.FULL: 0, MAMBA: 0}
    component._evict_device_start(1)
    node_id = component._evict_device_next_node(tracker, device_frees, host_frees)
    component._evict_device_end()
    return node_id, [int(v.item()) for v in device_frees[MAMBA]]


class TestMambaEvictionThinning(CustomTestCase):
    def test_uniform_chain_is_thinned_to_alternate_states(self):
        """Regression for #36935: under pool pressure a cold chain used to lose
        its states shallow-first, so a branch matching its prefix found no
        checkpoint. Thinning drops every other state instead, so the survivors
        stay spread along the chain."""
        component, nodes, core = _build_chain([1] * 8)

        freed = [_evict_one(component)[1] for _ in range(4)]

        self.assertEqual(freed, [[1], [3], [5], [7]])
        survivors = [n for n in nodes if n.component_data[MAMBA].value is not None]
        self.assertEqual(survivors, [nodes[1], nodes[3], nodes[5], nodes[7]])
        self.assertEqual(core.component_evictable_size_[MAMBA], 4)
        self.assertTrue(
            all(
                node.component_data[ComponentType.FULL].value is not None
                for node in nodes
            )
        )

    def test_victim_minimizes_the_merged_gap(self):
        """Depths 4, 5, 6, 10: removing the node at 5 merges a gap of 2, the
        others 5, so the tail (depth 4) is spared although it is LRU-oldest."""
        component, nodes, core = _build_chain([4, 1, 1, 4])

        _, freed = _evict_one(component)

        self.assertEqual(freed, [2])
        self.assertIsNotNone(nodes[0].component_data[MAMBA].value)
        self.assertEqual(core.cascaded, [nodes[1]])

    def test_locked_holder_is_a_boundary_not_a_victim(self):
        component, nodes, core = _build_chain([4, 1, 1, 4])
        nodes[1].component_data[MAMBA].lock_ref = 1

        _, freed = _evict_one(component)

        # With depth 5 pinned, depths 4 and 6 tie at a gap of 5; the tail wins.
        self.assertEqual(freed, [1])
        self.assertIsNotNone(nodes[1].component_data[MAMBA].value)

    def test_session_referenced_holder_is_never_thinned(self):
        component, nodes, core = _build_chain([4, 1, 1, 4])
        nodes[1].component_data[MAMBA].session_ref = 1

        _, freed = _evict_one(component)

        self.assertEqual(freed, [1])
        self.assertIsNotNone(nodes[1].component_data[MAMBA].value)

    def test_fork_below_the_tail_ends_the_chain(self):
        """A fork holder bounds the chain and keeps its own state; the tail is
        the only candidate, so it is evicted as before."""
        component, nodes, core = _build_chain([4, 1, 1, 4])
        sibling = UnifiedTreeNode(core.tree_components)
        sibling.parent = nodes[1]
        sibling.key = RadixKey(token_ids=array("q", [99]))
        nodes[1].children["fork"] = sibling

        _, freed = _evict_one(component)

        self.assertEqual(freed, [1])
        self.assertIsNotNone(nodes[1].component_data[MAMBA].value)

    def test_ancestor_holder_bounds_the_tail_gap(self):
        """Depths 1, 2, 3 under a hot holder at depth 1: the tail at 2 has gap
        3 - 1 = 2 and the node at 3 has gap 4 - 2 = 2, so the tie keeps the
        shallow-first order; without the ancestor bound the tail's gap would
        read 3 and the deeper node would be dropped instead."""
        component, nodes, core = _build_chain([1, 1, 1, 1])
        # Refresh the shallowest node so the LRU tail is the second one.
        core.lru_lists[MAMBA].reset_node_mru(nodes[0])

        _, freed = _evict_one(component)

        self.assertEqual(freed, [2])

    def test_state_matched_by_another_request_is_never_thinned(self):
        """Regression for the shared-prefix hit-rate drop: a group's boundary
        state is refreshed to MRU by every member's match, but a geometry-only
        victim rule could still thin it because a member's private tail sits
        right below it. A MATCH_END on a non-MRU node marks the state reused,
        and reused states are boundaries for thinning."""
        component, nodes, core = _build_chain([4, 1, 1, 4])
        # A later request matches depth 5 (not the MRU node, which is depth 10).
        component.refresh_lru(LRURefreshPhase.MATCH_END, nodes[1], core.root_node)
        self.assertTrue(nodes[1].component_data[MAMBA].metadata[MAMBA_REUSED_KEY])

        _, freed = _evict_one(component)

        # Depth 5 would be the min-gap victim (2 vs 5); it is spared, and the
        # remaining candidates tie so the tail goes as before.
        self.assertEqual(freed, [1])
        self.assertIsNotNone(nodes[1].component_data[MAMBA].value)

    def test_inserter_re_match_on_the_mru_node_does_not_mark_reuse(self):
        """cache_unfinished_req re-matches the node it just inserted; that
        match must not protect the state or thinning would never apply to an
        in-flight chain."""
        component, nodes, core = _build_chain([4, 1, 1, 4])
        component.refresh_lru(LRURefreshPhase.MATCH_END, nodes[-1], core.root_node)

        self.assertNotIn(MAMBA_REUSED_KEY, nodes[-1].component_data[MAMBA].metadata)
        self.assertEqual(_evict_one(component)[1], [2])

    def test_device_leaf_at_the_tail_is_still_deleted_as_a_leaf(self):
        component, nodes, core = _build_chain([1, 1])
        core.evictable_device_leaves.add(nodes[1])
        core.lru_lists[MAMBA].reset_node_mru(nodes[0])

        node_id, freed = _evict_one(component)

        self.assertEqual(node_id, nodes[1].id)
        self.assertEqual(freed, [])


if __name__ == "__main__":
    unittest.main()
