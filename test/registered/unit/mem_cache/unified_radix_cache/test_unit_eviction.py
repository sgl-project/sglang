"""CPU-only unit tests for --enable-mamba-unit-eviction.

A cached Mamba state goes with the full-attention KV that only it can serve,
and cached KV that no state can serve is evicted before any other. Without the
switch an interior state goes alone and its KV stays.
"""

import itertools
import unittest
from array import array
from dataclasses import replace
from unittest.mock import patch

from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


class _Tree:
    """A FULL + MAMBA unified radix cache over CPU tensors, page size 1."""

    def __init__(
        self, unit_eviction, *, kv_size=256, mamba_cache_size=64, checkpoints_level=None
    ):
        cfg = CacheConfig(
            components=(FULL, MAMBA),
            enable_mamba_unit_eviction=unit_eviction,
            kv_size=kv_size,
            mamba_cache_size=mamba_cache_size,
            max_num_reqs=1024,
        )
        if checkpoints_level is not None:
            cfg = replace(cfg, hybrid_cache_checkpoints_level=checkpoints_level)
        # build_fixture defaults to an accelerator, but these tests use CPU tensors.
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            self.cache, self.allocator, self.pool = build_fixture(cfg)
        self.core = self.cache.tree_core
        self.kv_capacity = self.allocator.available_size()
        self.rid = 0

    def insert(self, tokens, *, state=True):
        value = self.allocator.alloc(len(tokens))
        assert value is not None, "the test's KV pool is exhausted"
        params = InsertParams(key=RadixKey(array("q", tokens)), value=value)
        if state:
            self.rid += 1
            req = Req(
                rid=f"unit-eviction-{self.rid}",
                origin_input_text="",
                origin_input_ids=array("q"),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            acquire_rows(self.pool, self.pool.state, [req])
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        else:
            params.allow_missing_mamba = True
        self.cache.insert(params)

    def node(self, tokens):
        """The node whose key ends right after ``tokens``, or None."""
        node, depth = self.core.root_node, 0
        while depth < len(tokens):
            for child in node.children.values():
                key = list(child.key.raw_token_ids())
                if tokens[depth : depth + len(key)] == key:
                    node, depth = child, depth + len(key)
                    break
            else:
                return None
        return node

    def has_kv(self, tokens):
        node = self.node(tokens)
        return node is not None and node.component_data[FULL].value is not None

    def has_state(self, tokens):
        node = self.node(tokens)
        return node is not None and node.component_data[MAMBA].value is not None

    def tree_tokens(self):
        return len(self.core.all_values_flatten())

    def check(self, test):
        self.cache.sanity_check()
        test.assertEqual(
            self.allocator.available_size() + self.tree_tokens(),
            self.kv_capacity,
            "KV slots leaked or double-freed",
        )


A, B, C, D = [1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 2, 7, 8]


class TestMambaUnitEviction(CustomTestCase):
    def test_interior_state_without_states_below_takes_its_kv(self):
        """An interior state whose subtree holds no other state goes with the
        KV of that subtree: no request can resume inside it."""
        for unit_eviction in (False, True):
            with self.subTest(unit_eviction=unit_eviction):
                tree = _Tree(unit_eviction)
                tree.insert(A)
                tree.insert(B, state=False)
                result = tree.cache.evict(EvictParams(mamba_num=1))
                self.assertEqual(result.mamba_num_evicted, 1)
                self.assertFalse(tree.has_state(A))
                if unit_eviction:
                    self.assertIsNone(tree.node(A))
                    self.assertEqual(tree.tree_tokens(), 0)
                else:
                    # Upstream: evicting an interior state does not cascade.
                    self.assertTrue(tree.has_kv(A))
                    self.assertTrue(tree.has_kv(B))
                tree.check(self)

    def test_leaf_state_takes_the_kv_only_it_needs(self):
        """Evicting the last state of a chain releases the stateless ancestors
        above it, up to the nearest one whose subtree still holds a state."""
        for level, unit_eviction in itertools.product((2, 3), (False, True)):
            with self.subTest(level=level, unit_eviction=unit_eviction):
                tree = _Tree(unit_eviction, checkpoints_level=level)
                tree.insert(A)
                tree.insert(B)
                tree.insert(C)
                # The two interior states go first, each leaving KV that the
                # state below still needs: by LRU, or at level 2 to the path
                # cap as each deeper state is cached.
                if level == 3:
                    tree.cache.evict(EvictParams(mamba_num=2))
                self.assertFalse(tree.has_state(A))
                self.assertFalse(tree.has_state(B))
                self.assertTrue(tree.has_kv(A) and tree.has_kv(B))
                tree.check(self)
                tree.cache.evict(EvictParams(mamba_num=1))
                self.assertIsNone(tree.node(C))
                if unit_eviction:
                    self.assertEqual(tree.tree_tokens(), 0)
                else:
                    self.assertTrue(tree.has_kv(A) and tree.has_kv(B))
                tree.check(self)

    def test_a_release_takes_only_the_kv_its_state_served(self):
        """A state on a sibling branch keeps the shared prefix, and KV below a
        state that remains waits for the token walk."""
        tree = _Tree(unit_eviction=True)
        tree.insert(A, state=False)
        tree.insert(B)
        tree.insert(C, state=False)
        tree.insert(D)
        tree.insert(D + [9], state=False)
        released = [tree.node(B), tree.node(C)]
        tree.core.lru_lists[MAMBA].reset_node_mru(tree.node(D))
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertIsNone(tree.node(B))
        self.assertTrue(tree.has_kv(A), "D's state still needs the shared prefix")
        self.assertTrue(tree.has_state(D))
        self.assertTrue(tree.has_kv(D + [9]))
        # A walk that still holds a deleted node must see it evicted.
        self.assertTrue(all(node.evicted for node in released))
        tree.check(self)

    def test_a_locked_node_keeps_its_kv_and_its_ancestors(self):
        tree = _Tree(unit_eviction=True)
        tree.insert(A)
        tree.insert(B, state=False)
        locked = tree.node(B)
        receipt = tree.core.inc_lock_ref(locked.id)
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertFalse(tree.has_state(A))
        self.assertTrue(tree.has_kv(A) and tree.has_kv(B))
        tree.check(self)
        tree.core.dec_lock_ref(locked.id, receipt.to_dec_params())
        # Released, it is the first KV a token eviction takes.
        tree.insert([9, 9])
        tree.cache.evict(EvictParams(num_tokens=2))
        self.assertIsNone(tree.node(B))
        self.assertTrue(tree.has_state([9, 9]))
        tree.check(self)

    def test_kv_no_state_serves_is_evicted_first(self):
        """The token walk takes KV that no state serves before an older leaf
        that holds a state; without the switch it goes by recency alone."""
        for unit_eviction in (False, True):
            with self.subTest(unit_eviction=unit_eviction):
                tree = _Tree(unit_eviction)
                tree.insert([5, 6])
                tree.insert([7, 8], state=False)
                tree.cache.evict(EvictParams(num_tokens=2))
                if unit_eviction:
                    self.assertTrue(tree.has_state([5, 6]))
                    self.assertIsNone(tree.node([7, 8]))
                else:
                    self.assertIsNone(tree.node([5, 6]))
                    self.assertTrue(tree.has_kv([7, 8]))
                tree.check(self)

    def test_server_args_refuse_what_it_does_not_support(self):
        for flag, kwargs in (
            ("--disable-radix-cache", dict(disable_radix_cache=True)),
            ("--enable-hierarchical-cache", dict(enable_hierarchical_cache=True)),
            ("--radix-cache-backend", dict(radix_cache_backend="custom")),
        ):
            with self.subTest(flag=flag):
                args = ServerArgs(
                    model_path="dummy", enable_mamba_unit_eviction=True, **kwargs
                )
                with self.assertRaisesRegex(ValueError, flag):
                    handle_mamba_backend(args)


if __name__ == "__main__":
    unittest.main()
