"""CPU-only unit tests for --enable-mamba-unit-eviction.

A cached Mamba state goes with the full-attention KV that only it can serve,
and cached KV that no state can serve is evicted before any other. Without the
switch an interior state goes alone and its KV stays.
"""

import random
import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    resolve_tree_core_backend,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


class _Tree:
    """A FULL + MAMBA unified radix cache over CPU tensors, page size 1."""

    def __init__(self, unit_eviction, *, kv_size=256, mamba_cache_size=64):
        cfg = CacheConfig(
            components=(FULL, MAMBA),
            enable_mamba_unit_eviction=unit_eviction,
            kv_size=kv_size,
            mamba_cache_size=mamba_cache_size,
            max_num_reqs=1024,
        )
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
        for unit_eviction in (False, True):
            with self.subTest(unit_eviction=unit_eviction):
                tree = _Tree(unit_eviction)
                tree.insert(A)
                tree.insert(B)
                tree.insert(C)
                # The two interior states go first (LRU), each leaving KV that
                # the state below still needs.
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

    def test_interior_release_climbs_to_the_nearest_state_holder(self):
        tree = _Tree(unit_eviction=True)
        tree.insert([5, 6])
        tree.insert(A, state=False)
        tree.insert(B)
        tree.insert(C, state=False)
        # [5, 6] is the older state but a leaf; make B the LRU state.
        tree.core.lru_lists[MAMBA].reset_node_mru(tree.node([5, 6]))
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertIsNone(tree.node(A), "A's subtree holds no state any more")
        self.assertTrue(tree.has_state([5, 6]))
        self.assertEqual(tree.tree_tokens(), 2)
        tree.check(self)

    def test_interior_release_stops_below_a_state_on_a_sibling_branch(self):
        tree = _Tree(unit_eviction=True)
        tree.insert(A, state=False)
        tree.insert(B)
        tree.insert(C, state=False)
        tree.insert(D)
        released = [tree.node(B), tree.node(C)]
        tree.core.lru_lists[MAMBA].reset_node_mru(tree.node(D))
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertIsNone(tree.node(B))
        self.assertTrue(tree.has_kv(A), "D's state still needs the shared prefix")
        self.assertTrue(tree.has_state(D))
        # A walk that still holds a deleted node must see it evicted.
        self.assertTrue(all(node.evicted for node in released))
        tree.check(self)

    def test_a_release_takes_only_the_kv_its_state_served(self):
        """KV that no state serves elsewhere, beside the released unit or
        below a state that remains, waits for the token walk."""
        tree = _Tree(unit_eviction=True)
        tree.insert(A, state=False)
        tree.insert(B)
        tree.insert(C, state=False)
        tree.insert(D)
        tree.insert(D + [9], state=False)
        tree.core.lru_lists[MAMBA].reset_node_mru(tree.node(D))
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertIsNone(tree.node(B))
        self.assertTrue(tree.has_kv(D + [9]))
        tree.check(self)

        tree = _Tree(unit_eviction=True)
        tree.insert(A)
        tree.insert(B)
        tree.insert(D, state=False)
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertFalse(tree.has_state(A))
        self.assertTrue(tree.has_kv(D))
        tree.check(self)

    def test_a_state_cached_where_a_key_split_counts(self):
        """A state cached on an existing node, where an earlier insert split a
        key, keeps the KV above it when the deeper state goes."""
        tree = _Tree(unit_eviction=True)
        tree.insert(C)
        tree.insert(B, state=False)
        tree.insert(B)
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertIsNone(tree.node(C))
        self.assertTrue(tree.has_state(B))
        self.assertEqual(tree.tree_tokens(), len(B))
        tree.check(self)

    def test_a_host_copy_of_the_state_keeps_the_kv(self):
        tree = _Tree(unit_eviction=True)
        tree.insert(A)
        tree.insert(B, state=False)
        node = tree.node(A)
        state = node.component_data[MAMBA].value
        tree.core.set_component_host_value_raw(node.id, MAMBA, state.clone())
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertFalse(tree.has_state(A))
        self.assertTrue(tree.has_kv(A) and tree.has_kv(B))

    def test_a_state_on_another_branch_stops_the_release(self):
        tree = _Tree(unit_eviction=True)
        tree.insert(A, state=False)
        tree.insert(B)
        tree.insert(D)
        # B is the older leaf state.
        tree.cache.evict(EvictParams(mamba_num=1))
        self.assertIsNone(tree.node(B))
        self.assertTrue(tree.has_kv(A), "D's state still needs the shared prefix")
        self.assertTrue(tree.has_state(D))
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

    def test_kv_no_state_serves_goes_least_recent_first(self):
        for reused in (False, True):
            with self.subTest(reused=reused):
                tree = _Tree(unit_eviction=True)
                tree.insert([5, 6])
                tree.insert([7, 8], state=False)
                tree.insert([9, 9], state=False)
                if reused:
                    tree.insert([7, 8], state=False)  # cached again: used last
                tree.cache.evict(EvictParams(num_tokens=2))
                self.assertEqual(tree.node([7, 8]) is None, not reused)
                self.assertEqual(tree.node([9, 9]) is None, reused)
                self.assertTrue(tree.has_state([5, 6]))
                tree.check(self)

    def test_kv_no_state_serves_goes_first_once_a_request_releases_it(self):
        tree = _Tree(unit_eviction=True)
        tree.insert([5, 6])
        tree.insert([3, 3])
        tree.insert([7, 8], state=False)
        held = tree.node([7, 8]).id
        receipt = tree.core.inc_lock_ref(held)
        tree.cache.evict(EvictParams(num_tokens=2))
        self.assertIsNone(tree.node([5, 6]))  # the least recent leaf instead
        tree.core.dec_lock_ref(held, receipt.to_dec_params())
        tree.cache.evict(EvictParams(num_tokens=2))
        self.assertIsNone(tree.node([7, 8]))
        self.assertTrue(tree.has_state([3, 3]))
        tree.check(self)

    def test_the_queue_of_kv_no_state_serves_stays_bounded(self):
        """Each use of KV that no state serves queues it again; stale entries
        are dropped once they outnumber the evictable leaves."""
        tree = _Tree(unit_eviction=True)
        tree.insert([5, 6])
        tree.insert([7, 8], state=False)
        for _ in range(500):
            tree.insert([7, 8], state=False)
            held = tree.node([7, 8]).id
            tree.core.dec_lock_ref(held, tree.core.inc_lock_ref(held).to_dec_params())
        self.assertLessEqual(len(tree.core.state_units._dead), 2 * 2 + 64)
        tree.cache.evict(EvictParams(num_tokens=2))
        self.assertIsNone(tree.node([7, 8]))
        self.assertTrue(tree.has_state([5, 6]))
        tree.check(self)
        tree.insert([7, 8], state=False)
        tree.cache.reset()  # a flush drops the old tree's nodes from the queue
        self.assertEqual(len(tree.core.state_units._dead), 0)

    def test_the_sanity_check_reports_drifted_counts_and_unqueued_kv(self):
        tree = _Tree(unit_eviction=True)
        tree.insert(C)
        tree.insert([7, 8], state=False)
        tree.cache.sanity_check()
        tree.node(C)._unit_states_below += 1
        with self.assertRaisesRegex(AssertionError, r"\[Units\] node \d+ counts"):
            tree.cache.sanity_check()
        tree.node(C)._unit_states_below -= 1
        tree.core.state_units._dead_entry.clear()
        with self.assertRaisesRegex(AssertionError, "not queued"):
            tree.cache.sanity_check()

    def test_the_switch_runs_on_the_python_tree_core(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            enable_mamba_unit_eviction=True,
        )
        self.assertEqual(resolve_tree_core_backend("rust", params), "python")

    def test_server_args_refuse_it_without_the_radix_cache(self):
        from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend

        args = ServerArgs(
            model_path="dummy",
            enable_mamba_unit_eviction=True,
            disable_radix_cache=True,
        )
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            handle_mamba_backend(args)

    def test_server_args_refuse_it_with_hierarchical_cache(self):
        from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend

        args = ServerArgs(
            model_path="dummy",
            enable_mamba_unit_eviction=True,
            enable_hierarchical_cache=True,
        )
        with self.assertRaisesRegex(ValueError, "--enable-hierarchical-cache"):
            handle_mamba_backend(args)

    def test_server_args_refuse_it_with_a_registered_radix_cache_backend(self):
        from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend

        args = ServerArgs(
            model_path="dummy",
            enable_mamba_unit_eviction=True,
            radix_cache_backend="custom",
        )
        with self.assertRaisesRegex(ValueError, "--radix-cache-backend"):
            handle_mamba_backend(args)


class TestMambaUnitEvictionRandomWorkload(CustomTestCase):
    """Sessions that extend, fork and finish under memory pressure: the tree
    stays consistent after every step, no KV slot leaks, and a token eviction
    never takes a state while KV that no state serves is left."""

    def _run(self, unit_eviction, seed):
        rng = random.Random(seed)
        tree = _Tree(unit_eviction, kv_size=160, mamba_cache_size=48)
        sessions = [[s * 1000] for s in range(6)]
        for step in range(250):
            op = rng.random()
            if op < 0.7:
                i = rng.randrange(len(sessions))
                if rng.random() < 0.15:
                    other = sessions[rng.randrange(len(sessions))]
                    sessions[i] = other[: rng.randint(1, len(other))]
                grow = [rng.randrange(1, 900) for _ in range(rng.randint(1, 6))]
                sessions[i] = sessions[i] + grow
                if len(sessions[i]) > 48:
                    sessions[i] = [i * 1000 + step]
                need = len(sessions[i])
                if tree.allocator.available_size() < need:
                    tree.cache.evict(EvictParams(num_tokens=need))
                if tree.allocator.available_size() < need:
                    continue
                if tree.cache._pool_available(MAMBA) < 1:
                    tree.cache.evict(EvictParams(mamba_num=1))
                tree.insert(sessions[i], state=rng.random() < 0.85)
            elif op < 0.85:
                tree.cache.evict(EvictParams(mamba_num=rng.randint(1, 3)))
            else:
                states_before = tree.core.mamba_evictable_size()
                dead = self._dead_leaf_tokens(tree)
                want = rng.randint(1, 8)
                tree.cache.evict(EvictParams(num_tokens=want))
                if unit_eviction and dead >= want:
                    self.assertEqual(
                        tree.core.mamba_evictable_size(),
                        states_before,
                        f"step {step}: a state went before dead KV",
                    )
            tree.check(self)

    @staticmethod
    def _dead_leaf_tokens(tree):
        """Tokens of the leaves that no state at or below serves, whose KV
        goes first."""

        def serves(node):
            stack = [node]
            while stack:
                cur = stack.pop()
                cd = cur.component_data[MAMBA]
                if cd.value is not None or cd.host_value is not None:
                    return True
                stack.extend(cur.children.values())
            return False

        return sum(
            len(node.component_data[FULL].value)
            for node in tree.core.evictable_device_leaves
            if not serves(node)
        )

    def test_random_workload_with_unit_eviction(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self._run(unit_eviction=True, seed=seed)

    def test_random_workload_without_unit_eviction(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self._run(unit_eviction=False, seed=seed)


if __name__ == "__main__":
    unittest.main()
