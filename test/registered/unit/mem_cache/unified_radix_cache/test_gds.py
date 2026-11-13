"""CPU-only unit tests for --radix-eviction-policy gds: GreedyDual-Size
eviction of a hybrid model's cached states, each with the KV only it serves."""

import random
import unittest
from array import array
from dataclasses import replace
from unittest.mock import patch

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.utils import get_eviction_strategy
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


class _Tree:
    """A FULL + MAMBA tree under gds over CPU tensors, page size 1. With the
    default ``prefill_cost`` [1, 0], a unit's gain is the tokens since the
    state above it; a state slot counts as ``state_tokens`` KV tokens. Tests
    that cache states at several depths of one path pass
    ``checkpoints_level=3``: level 2 keeps one state per path."""

    def __init__(self, kv_size=256, state_tokens=10, checkpoints_level=None, **config):
        cfg = CacheConfig(
            components=(FULL, MAMBA),
            kv_size=kv_size,
            mamba_cache_size=64,
            max_num_reqs=1024,
            eviction_policy="gds",
            eviction_policy_config=config or None,
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
        self.core.gds.state_tokens = state_tokens
        self.kv_capacity = self.allocator.available_size()
        self.rid = 0

    def insert(self, tokens, *, state=True):
        value = self.allocator.alloc(len(tokens))
        assert value is not None, "the test's KV pool is exhausted"
        params = InsertParams(key=RadixKey(array("q", tokens)), value=value)
        if state:
            self.rid += 1
            req = Req(
                rid=f"gds-{self.rid}",
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

    def has_state(self, tokens):
        node = self.node(tokens)
        return node is not None and node.component_data[MAMBA].value is not None

    def evict_states(self, n=1):
        return self.cache.evict(EvictParams(mamba_num=n))

    def check(self, test):
        self.cache.sanity_check()
        test.assertEqual(
            self.allocator.available_size() + len(self.core.all_values_flatten()),
            self.kv_capacity,
        )


A, B = [1, 2, 3, 4], list(range(10, 20))


class TestGreedyDualSizeOrder(CustomTestCase):
    def test_the_lowest_gain_per_size_goes_first(self):
        tree = _Tree()
        tree.insert(A)  # gain 4, size 10 + 4
        tree.insert(B)  # gain 10, size 10 + 10
        tree.evict_states()
        self.assertIsNone(tree.node(A))
        self.assertTrue(tree.has_state(B))
        tree.check(self)

    def test_gain_counts_from_the_nearest_state_above(self):
        tree = _Tree(checkpoints_level=3)
        tree.insert(list(range(1, 7)))
        tree.insert(list(range(1, 9)))  # gain 2 over the state at 6, size 12
        tree.insert([20, 21, 22])  # gain 3, size 13
        tree.evict_states()
        self.assertFalse(tree.has_state(list(range(1, 9))))
        self.assertTrue(tree.has_state([20, 21, 22]))

    def test_size_counts_the_kv_only_the_unit_serves(self):
        tree = _Tree(checkpoints_level=3)
        tree.insert([1, 2, 3, 4])  # interior: gain 4, size 10
        tree.insert(list(range(1, 13)) + [99])  # leaf
        tree.insert(list(range(30, 36)))  # gain 6, size 10 + 6
        tree.evict_states()
        self.assertFalse(tree.has_state(list(range(30, 36))))
        self.assertTrue(tree.has_state([1, 2, 3, 4]))

    def test_interior_states_go_before_leaves_with_leaf_last(self):
        for leaf_last in (False, True):
            with self.subTest(leaf_last=leaf_last):
                tree = _Tree(leaf_last=leaf_last, checkpoints_level=3)
                tree.insert([1, 2])  # interior once its child is cached: 2 / 10
                tree.insert([1, 2, 3, 4, 5, 6])  # leaf: 4 / 14
                tree.insert([7])  # leaf: 1 / 11
                tree.evict_states()
                self.assertEqual(tree.has_state([1, 2]), not leaf_last)
                self.assertEqual(tree.has_state([7]), leaf_last)
                tree.check(self)

    def test_a_hit_refreshes_the_stamp(self):
        for hit in (None, "match", "insert"):
            with self.subTest(hit=hit):
                tree = _Tree()
                tree.insert([50])  # the lowest score: it goes first
                tree.insert([7, 8, 9, 10])
                tree.evict_states()  # raises the clock
                tree.insert([5, 6, 7, 8])  # same gain and size, newer stamp
                if hit == "match":
                    tree.cache.match_prefix(
                        MatchPrefixParams(key=RadixKey(array("q", [7, 8, 9, 10, 11])))
                    )
                elif hit == "insert":
                    tree.insert([7, 8, 9, 10])  # the same state cached again
                tree.evict_states()
                self.assertEqual(tree.has_state([7, 8, 9, 10]), hit is not None)
                self.assertEqual(tree.has_state([5, 6, 7, 8]), hit is None)

    def test_a_locked_unit_is_passed_over_and_kept(self):
        tree = _Tree()
        tree.insert(A)
        tree.insert(B)
        receipt = tree.core.inc_lock_ref(tree.node(A).id)
        tree.evict_states()
        self.assertTrue(tree.has_state(A))
        self.assertFalse(tree.has_state(B))
        tree.core.dec_lock_ref(tree.node(A).id, receipt.to_dec_params())
        tree.evict_states()
        self.assertFalse(tree.has_state(A))
        tree.check(self)

    def test_a_token_request_takes_only_units_that_free_kv(self):
        tree = _Tree(checkpoints_level=3)
        tree.insert([1, 2])  # interior: frees no KV, the lowest score
        tree.insert(list(range(1, 11)))
        tree.insert(B)
        result = tree.cache.evict(EvictParams(num_tokens=1))
        self.assertGreaterEqual(result.num_tokens_evicted, 1)
        self.assertTrue(tree.has_state([1, 2]))
        tree.check(self)

    def test_random_workload_keeps_every_heap_entry_at_its_score(self):
        """The sanity check also holds every unit's heap entry to its score.
        With a token at position p costing 1 + p and a state slot counting as
        one token, the path cap lowers the scores of the units below it."""
        for leaf_last, e, state_tokens, path_cap in (
            (False, 0, 8, -1),
            (True, 1e5, 1, 1),
        ):
            with self.subTest(leaf_last=leaf_last, path_cap=path_cap):
                self._run(0, leaf_last, e, state_tokens, path_cap)

    def _run(self, seed, leaf_last, e, state_tokens, path_cap):
        rng = random.Random(seed)
        tree = _Tree(
            kv_size=160,
            state_tokens=state_tokens,
            leaf_last=leaf_last,
            prefill_cost=[1, e],
        )
        tree.cache.components[MAMBA].mamba_max_states_per_path = path_cap
        sessions = [[s * 1000] for s in range(6)]
        locks = []
        for step in range(250):
            op = rng.random()
            if op < 0.65:
                i = rng.randrange(len(sessions))
                if rng.random() < 0.15:
                    other = sessions[rng.randrange(len(sessions))]
                    sessions[i] = other[: rng.randint(1, len(other))]
                sessions[i] = sessions[i] + [
                    rng.randrange(1, 900) for _ in range(rng.randint(1, 6))
                ]
                if len(sessions[i]) > 48:
                    sessions[i] = [i * 1000 + step]
                need = len(sessions[i])
                if tree.allocator.available_size() < need:
                    tree.cache.evict(EvictParams(num_tokens=need))
                if tree.allocator.available_size() < need:
                    continue
                if tree.cache._pool_available(MAMBA) < 1:
                    tree.evict_states()
                tree.insert(sessions[i], state=rng.random() < 0.85)
            elif op < 0.72:
                tokens = sessions[rng.randrange(len(sessions))]
                node_id = tree.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens)))
                ).last_device_node
                if node_id != tree.core.root_node.id:
                    locks.append((node_id, tree.core.inc_lock_ref(node_id)))
            elif op < 0.8 and locks:
                node_id, receipt = locks.pop(rng.randrange(len(locks)))
                tree.core.dec_lock_ref(node_id, receipt.to_dec_params())
            elif op < 0.9:
                tree.evict_states(rng.randint(1, 3))
            else:
                tree.cache.evict(EvictParams(num_tokens=rng.randint(1, 8)))
            tree.check(self)

    def test_unsupported_configurations_are_refused(self):
        self.assertEqual(get_eviction_strategy("gds").prefill_cost, (1.0, 0.0))
        for config, error in (
            ({"prefill_cost": [1]}, ValueError),
            ({"prefill_cost": [1, -1]}, ValueError),
            ({"leaflast": True}, TypeError),
        ):
            with self.subTest(config=config), self.assertRaises(error):
                get_eviction_strategy("gds", config)
        for flag, kwargs in (
            ("disable-radix-cache", dict(disable_radix_cache=True)),
            ("radix-cache-backend", dict(radix_cache_backend="custom")),
            ("hierarchical", dict(enable_hierarchical_cache=True)),
            ("session", dict(enable_session_radix_cache=True)),
        ):
            with self.subTest(flag=flag):
                args = ServerArgs(
                    model_path="dummy", radix_eviction_policy="gds", **kwargs
                )
                with self.assertRaisesRegex(ValueError, flag):
                    handle_cache_compatibility(args)


if __name__ == "__main__":
    unittest.main()
