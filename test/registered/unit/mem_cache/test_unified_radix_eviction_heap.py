"""Persistent lazy eviction heap for the UnifiedRadixCache Full component.

Covers ``_LazyLeafHeap`` directly and proves that eviction order is identical
to the legacy per-call heap rebuild by replaying the same randomized op stream
on two caches: one running the new code and one whose Full component carries
the legacy ``_evict_device_*`` / ``drive_host_eviction`` bodies (installed
through ``component_registry_override``).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import heapq
import random
import unittest
from array import array
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import test_write_back_eviction_parity as write_back_parity
import torch
from numpy import float64

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components import base as _tree_component
from sglang.srt.mem_cache.unified_cache.components.full import (
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import _LazyLeafHeap
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test import hicache_write_back_fixtures
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of
from sglang.test.unified_allocator_fixtures import build_window_chain

POLICIES = ("lru", "lfu", "fifo", "mru", "filo", "priority", "slru")


# ---------------------------------------------------------------------------
# Legacy oracle: the Full component as it was before the persistent heap.
# ---------------------------------------------------------------------------
class LegacyFullComponent(FullComponent):
    """Verbatim pre-change eviction drivers (rebuild the heap every walk)."""

    def __init__(self, cache, params, slot):
        super().__init__(cache, params, slot)
        self.session_ref_eviction_strategy = (
            self._session_ref_eviction_strategy
            if cache.enable_session_radix_cache
            else None
        )
        self._host_eviction_heap: Optional[list] = None

    def _ensure_eviction_strategy(self) -> None:
        if self.session_ref_eviction_strategy is None:
            self.session_ref_eviction_strategy = (
                self.tree_core.eviction_strategy.get_priority
            )

    def _session_ref_eviction_strategy(self, node):
        ref = self.session_ref(node)
        return ref > 0, ref, self.tree_core.eviction_strategy.get_priority(node)

    def _evict_device_start(self, request_cnt: int) -> None:
        self._ensure_eviction_strategy()
        self._evict_device_request_cnt = request_cnt
        self._evict_device_last_node = None
        self._evict_device_heap = [
            (self.session_ref_eviction_strategy(n), n)
            for n in self.tree_core.evictable_device_leaves
        ]
        heapq.heapify(self._evict_device_heap)

    def _evict_device_next_node(self, tracker, device_frees, host_frees, cache_actions):
        ct = self.component_type
        lv = self._evict_device_last_node
        if (
            lv is not None
            and lv.parent is not None
            and self.tree_core._is_device_walk_candidate(lv.parent)
        ):
            heapq.heappush(
                self._evict_device_heap,
                (self.session_ref_eviction_strategy(lv.parent), lv.parent),
            )
        self._evict_device_last_node = None
        while tracker[ct] < self._evict_device_request_cnt and self._evict_device_heap:
            _, x = heapq.heappop(self._evict_device_heap)
            if not self.tree_core._is_device_walk_candidate(x):
                continue
            self._evict_device_last_node = x
            return x.id
        return None

    def _evict_device_end(self) -> None:
        self._evict_device_heap = []
        self._evict_device_last_node = None
        self._host_eviction_heap = None

    def on_host_leaf_added(self, node) -> None:
        if self._host_eviction_heap is not None:
            heapq.heappush(
                self._host_eviction_heap,
                (self.session_ref_eviction_strategy(node), node),
            )

    def drive_host_eviction(
        self, num_tokens, tracker, device_frees, host_frees, cache_actions
    ):
        self._ensure_eviction_strategy()
        if tracker[self.component_type] >= num_tokens:
            return
        if self._host_eviction_heap is None:
            self._host_eviction_heap = [
                (self.session_ref_eviction_strategy(n), n)
                for n in self.tree_core.evictable_host_leaves
            ]
            heapq.heapify(self._host_eviction_heap)
        heap = self._host_eviction_heap
        ct = self.component_type
        try:
            while tracker[ct] < num_tokens and heap:
                _, x = heapq.heappop(heap)
                if x not in self.tree_core.evictable_host_leaves:
                    continue
                self.tree_core._evict_host_leaf(
                    x, tracker, device_frees, host_frees, cache_actions=cache_actions
                )
        finally:
            if not (
                self.is_evict_device_ongoing
                and not self.tree_core.enable_session_radix_cache
            ):
                self._host_eviction_heap = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_cache(
    *,
    policy: str = "lru",
    kv_size: int = 512,
    enable_session: bool = False,
    legacy: bool = False,
    page_size: int = 1,
    policy_config: Optional[dict] = None,
) -> UnifiedRadixCache:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=kv_size,
        page_size=page_size,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=kv_size, dtype=dtype, device="cpu", kvcache=kv_pool, need_sort=False
    )
    req_pool = ReqToTokenPool(
        size=8, max_context_len=1024, device="cpu", enable_memory_saver=False
    )
    # The heap lives in the Python TreeCore; pin it so the suite does not pick
    # up the process-wide backend default.
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
        cache = UnifiedRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                eviction_policy=policy,
                eviction_policy_config=policy_config,
                enable_session_radix_cache=enable_session,
                tree_slots=tree_slots_of((ComponentType.FULL,)),
                component_registry_override=(
                    {TREE_TOKEN: LegacyFullComponent} if legacy else None
                ),
            )
        )
    if legacy:
        forward_host_leaf_adds(cache)
    return cache


class _HostLeafAddForwarder:
    """Stands in for the tree core's host heap on the legacy side: the legacy
    drivers never read it, but they need the host-leaf-added notification the
    tree core used to send to the component."""

    def __init__(self, heap, component):
        self._heap = heap
        self._component = component

    def add(self, node) -> None:
        self._heap.add(node)
        self._component.on_host_leaf_added(node)

    def __getattr__(self, name):
        return getattr(self._heap, name)


def forward_host_leaf_adds(cache) -> None:
    core = cache.tree_core
    core.full_host_heap = _HostLeafAddForwarder(
        core.full_host_heap, core.components_by_type[ComponentType.FULL]
    )


def node_path(core, node) -> tuple:
    """Root-to-node token path: identifies a node across two trees."""
    parts = []
    while node is not None and node is not core.root_node:
        parts.append(tuple(node.key.token_ids))
        node = node.parent
    return tuple(reversed(parts))


def reset_time_counter() -> None:
    _tree_component._LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)


def gen_sequences(rng: random.Random, n: int, page_size: int) -> list[list[int]]:
    """Sequences with tree-like prefix sharing (chains + fan-out)."""
    seqs = [[rng.randint(1, 50) for _ in range(page_size)]]
    while len(seqs) < n:
        parent = rng.choice(seqs)
        tail = [rng.randint(1, 50) for _ in range(page_size * rng.randint(1, 6))]
        seqs.append(parent + tail)
    return seqs


def gen_ops(seed: int, num_ops: int, page_size: int, session: bool) -> list[tuple]:
    rng = random.Random(seed)
    seqs = gen_sequences(rng, 60, page_size)
    ops = []
    for _ in range(num_ops):
        r = rng.random()
        if r < 0.40:
            ops.append(("insert", rng.choice(seqs), rng.randint(0, 3)))
        elif r < 0.60:
            ops.append(("match", rng.choice(seqs)))
        elif r < 0.70:
            ops.append(("lock", rng.choice(seqs)))
        elif r < 0.78:
            ops.append(("unlock", rng.randint(0, 1 << 30)))
        elif session and r < 0.84:
            ops.append(("register", rng.choice(seqs), f"s{rng.randint(0, 3)}"))
        elif session and r < 0.88:
            ops.append(("release", f"s{rng.randint(0, 3)}"))
        else:
            ops.append(("evict", page_size * rng.randint(1, 6)))
    return ops


class Replay:
    """Drive one cache through an op stream, recording every eviction victim."""

    def __init__(self, cache: UnifiedRadixCache, check_every: int = 25):
        self.cache = cache
        self.core = cache.tree_core
        self.check_every = check_every
        self.victims: list[list[tuple]] = []
        self.evicted_counts: list[int] = []
        self._current: Optional[list] = None
        self._locked: list[tuple] = []
        orig = self.core.evict_device_leaf
        core = self.core

        def recording_evict_device_leaf(node_id, is_write_back):
            if self._current is not None:
                self._current.append(node_path(core, core.node_by_id(node_id)))
            return orig(node_id, is_write_back)

        self.core.evict_device_leaf = recording_evict_device_leaf

    def _alloc(self, n: int):
        alloc = self.cache.token_to_kv_pool_allocator
        v = alloc.alloc(n)
        if v is None:
            self.cache.evict(EvictParams(num_tokens=n * 2))
            v = alloc.alloc(n)
        return v

    def run(self, ops: list[tuple]) -> None:
        for i, op in enumerate(ops):
            kind = op[0]
            if kind == "insert":
                seq, prio = op[1], op[2]
                v = self._alloc(len(seq))
                if v is not None:
                    self.cache.insert(
                        InsertParams(
                            key=RadixKey(array("q", seq)),
                            value=v.to(torch.int64),
                            priority=prio,
                        )
                    )
            elif kind == "match":
                self.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", op[1])))
                )
            elif kind == "lock":
                res = self.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", op[1])))
                )
                node_id = res.last_device_node
                if node_id != self.cache.root_node_handle():
                    lr = self.cache.inc_lock_ref(node_id).to_dec_params()
                    self._locked.append((node_id, lr))
            elif kind == "unlock":
                if self._locked:
                    node_id, lr = self._locked.pop(op[1] % len(self._locked))
                    self.cache.dec_lock_ref(node_id, lr)
            elif kind == "register":
                seq, sid = op[1], op[2]
                res = self.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq)))
                )
                if res.last_device_node != self.cache.root_node_handle():
                    self.cache.session_refs.register_session_ref(
                        SimpleNamespace(
                            session_id=sid,
                            session_generation=self.cache.ensure_session_generation(
                                sid
                            ),
                            session=None,
                            last_node=res.last_device_node,
                            origin_input_ids=array("q", seq),
                            output_ids=array("q"),
                            extra_key=None,
                        ),
                        leaf=res.last_device_node,
                    )
            elif kind == "release":
                self.cache.release_radix_session(op[1])
            elif kind == "evict":
                self._current = []
                res = self.cache.evict(EvictParams(num_tokens=op[1]))
                self.victims.append(self._current)
                self.evicted_counts.append(res.num_tokens_evicted)
                self._current = None
            if i % self.check_every == 0:
                self.cache.sanity_check()
        # release everything so the final state is comparable
        while self._locked:
            node_id, lr = self._locked.pop()
            self.cache.dec_lock_ref(node_id, lr)
        self.cache.sanity_check()

    def leaf_paths(self) -> set:
        return {node_path(self.core, n) for n in self.core.evictable_device_leaves}


def replay_pair(
    policy: str,
    seed: int,
    session: bool,
    page_size: int = 1,
    policy_config: Optional[dict] = None,
):
    ops = gen_ops(seed, 320, page_size, session)
    pair = []
    for legacy in (False, True):
        reset_time_counter()
        replay = Replay(
            make_cache(
                policy=policy,
                enable_session=session,
                page_size=page_size,
                legacy=legacy,
                policy_config=policy_config,
            )
        )
        replay.run(ops)
        pair.append(replay)
    return pair


# ---------------------------------------------------------------------------
# _LazyLeafHeap unit tests
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("id", "key", "parent")

    def __init__(self, id, key):
        self.id = id
        self.key = key
        self.parent = None

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return f"N{self.id}"


def _drain(heap) -> list:
    heap.begin_walk()
    out = []
    while True:
        n = heap.pop_next()
        if n is None:
            break
        out.append(n)
    heap.end_walk()
    return out


class TestLazyLeafHeapUnit(CustomTestCase):
    def _heap(self, nodes, hide_entrants=True):
        members = set(nodes)
        heap = _LazyLeafHeap(members, lambda n: n.key, hide_entrants=hide_entrants)
        for n in nodes:
            heap.add(n)
        return members, heap

    def _assert_invariants(self, heap):
        errors = []
        heap.check_invariants(errors.append, "t", lambda n: True)
        self.assertEqual(errors, [])

    def test_pop_order_and_key_drop(self):
        nodes = [_Node(i, k) for i, k in enumerate([5, 3, 9, 1])]
        members, heap = self._heap(nodes)
        self._assert_invariants(heap)
        nodes[2].key = 0  # 9 -> 0: must come first once reported
        heap.lowered(nodes[2])
        self.assertEqual(_drain(heap), [nodes[2], nodes[3], nodes[1], nodes[0]])
        # a full walk yields every member exactly once, then they all have an entry again
        self.assertEqual(set(heap._entry), members)
        self._assert_invariants(heap)

    def test_key_raise_needs_no_hook(self):
        nodes = [_Node(i, k) for i, k in enumerate([1, 2, 3, 4])]
        members, heap = self._heap(nodes)
        before = heap.num_heap_entries()
        nodes[0].key = 10  # like an LRU touch: no call at all
        nodes[1].key = 7
        self.assertEqual(heap.num_heap_entries(), before)
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), [nodes[2], nodes[3], nodes[1], nodes[0]])
        self._assert_invariants(heap)

    def test_leaving_and_reentering_pushes_nothing(self):
        nodes = [_Node(i, i) for i in range(4)]
        members, heap = self._heap(nodes)
        before = heap.num_heap_entries()
        for t in range(1000):  # lock / unlock churn with LRU-style touches
            members.discard(nodes[0])
            nodes[0].key = 10 + t
            members.add(nodes[0])
            heap.add(nodes[0])
        self.assertEqual(heap.num_heap_entries(), before)
        self.assertEqual(_drain(heap), [nodes[1], nodes[2], nodes[3], nodes[0]])

    def test_non_member_entry_is_dropped_on_pop(self):
        nodes = [_Node(i, i) for i in range(4)]
        members, heap = self._heap(nodes)
        members.discard(nodes[0])  # left the set: no heap call
        self.assertEqual(_drain(heap), nodes[1:])
        self.assertNotIn(nodes[0], heap._entry)
        nodes[0].key = -1
        members.add(nodes[0])
        heap.add(nodes[0])
        self.assertEqual(_drain(heap), nodes)

    def test_key_drop_outside_the_set_drops_the_entry(self):
        nodes = [_Node(i, i + 1) for i in range(3)]
        members, heap = self._heap(nodes)
        members.discard(nodes[2])
        before = heap.num_heap_entries()
        for key in (0, -1, -2):  # e.g. MRU stamps on an interior node
            nodes[2].key = key
            heap.lowered(nodes[2])
        self.assertEqual(heap.num_heap_entries(), before)  # nothing pushed
        self.assertNotIn(nodes[2], heap._entry)
        members.add(nodes[2])
        heap.add(nodes[2])  # keyed again on the way back in
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), [nodes[2], nodes[0], nodes[1]])

    def test_lowered_ignores_nodes_without_an_entry(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        outsider = _Node(99, -1)
        heap.lowered(outsider)
        self.assertEqual(len(heap), 3)
        self.assertEqual(_drain(heap), nodes)

    def test_forget_drops_a_deleted_node(self):
        nodes = [_Node(i, i) for i in range(4)]
        members, heap = self._heap(nodes)
        members.discard(nodes[0])
        heap.forget(nodes[0])
        self.assertEqual(len(heap), 3)
        self.assertEqual(_drain(heap), nodes[1:])
        self._assert_invariants(heap)

    def test_walk_snapshot_entrants_and_promote(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        heap.begin_walk()
        first = heap.pop_next()
        self.assertIs(first, nodes[0])
        entrant = _Node(10, -5)  # smallest key, enters mid-walk
        members.add(entrant)
        heap.add(entrant)
        self.assertIs(heap.pop_next(), nodes[1])  # invisible without promote
        heap.promote(entrant)
        self.assertIs(heap.pop_next(), entrant)  # visible right after promote
        self.assertIs(heap.pop_next(), nodes[2])
        self.assertIsNone(heap.pop_next())
        heap.end_walk()
        # every member (including the yielded ones) has an entry again
        self.assertEqual(set(heap._entry), members)
        self._assert_invariants(heap)

    def test_hidden_entrant_with_an_old_entry_stays_hidden(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        members.discard(nodes[0])  # leaves before the walk, entry lingers
        heap.begin_walk()
        members.add(nodes[0])
        heap.add(nodes[0])  # re-enters mid-walk
        self.assertIs(heap.pop_next(), nodes[1])
        self.assertIs(heap.pop_next(), nodes[2])
        self.assertIsNone(heap.pop_next())
        heap.end_walk()
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), nodes)

    def test_host_walk_sees_entrants_at_once(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes, hide_entrants=False)
        heap.begin_walk()
        self.assertIs(heap.pop_next(), nodes[0])
        entrant = _Node(10, -5)
        members.add(entrant)
        heap.add(entrant)
        self.assertIs(heap.pop_next(), entrant)
        self.assertIs(heap.pop_next(), nodes[1])
        heap.end_walk()
        self._assert_invariants(heap)

    def test_promoted_non_member_lives_for_one_walk(self):
        # write_back: the walk may take a parent whose device children are all
        # queued although it is not in the leaf set.
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        parent = _Node(10, -1)
        heap.begin_walk()
        self.assertIs(heap.pop_next(), nodes[0])
        heap.promote(parent)
        self.assertIs(heap.pop_next(), parent)
        self.assertIs(heap.pop_next(), nodes[1])
        other = _Node(11, -2)
        heap.promote(other)  # promoted but never popped
        heap.end_walk()
        self.assertEqual(set(heap._entry), members)
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), nodes)

    def test_promoted_node_stays_visible_when_it_joins_the_set(self):
        # write_back with a full host pool: the queued batch is demoted in the
        # middle of the walk and the promoted parent becomes a real leaf.
        child, other, parent = _Node(0, 1), _Node(1, 2), _Node(2, 3)
        members, heap = self._heap([child, other])
        heap.begin_walk()
        self.assertIs(heap.pop_next(), child)
        heap.promote(parent)
        self.assertIs(heap.pop_next(), other)
        members.discard(child)
        members.add(parent)
        heap.add(parent)
        self.assertIs(heap.pop_next(), parent)
        heap.end_walk()
        self._assert_invariants(heap)

    def test_mid_walk_key_drop_keeps_the_walk_start_order(self):
        # A nested host eviction can lower a session_ref during the walk; the
        # heap rebuilt at walk start would not have seen it.
        x, y = _Node(0, 5), _Node(1, 8)
        members, heap = self._heap([x, y])
        x.key = 10  # raised before the walk, no hook: the entry is stale-low
        heap.begin_walk()
        heap.pin(x)
        x.key = 7
        heap.lowered(x)
        self.assertEqual([heap.pop_next(), heap.pop_next()], [y, x])
        heap.end_walk()
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), [x, y])  # the drop counts from now on

    def test_mid_walk_key_drop_is_applied_at_end_walk(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        heap.begin_walk()
        heap.pin(nodes[2])
        nodes[2].key = -100
        heap.lowered(nodes[2])
        self.assertIs(heap.pop_next(), nodes[0])
        heap.end_walk()
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), [nodes[2], nodes[0], nodes[1]])

    def test_host_walk_takes_an_entrant_at_its_current_key(self):
        a, b, p = _Node(0, 5), _Node(1, 8), _Node(2, 10)
        members, heap = self._heap([a, b, p], hide_entrants=False)
        members.discard(p)  # gained a host child: the entry lingers
        heap.begin_walk()
        heap.pin(p)
        p.key = 6  # the walk evicts that child and a session recedes
        heap.lowered(p)
        members.add(p)
        heap.add(p)
        self.assertEqual([heap.pop_next() for _ in range(3)], [a, p, b])
        heap.end_walk()

    def test_nested_walk_is_rejected(self):
        members, heap = self._heap([_Node(0, 0)])
        heap.begin_walk()
        with self.assertRaises(AssertionError):
            heap.begin_walk()
        heap.end_walk()

    def test_stale_entries_are_bounded_with_bounded_work_per_call(self):
        nodes = [_Node(i, 0) for i in range(8)]
        members, heap = self._heap(nodes)
        steps = _LazyLeafHeap._DRAIN_STEPS
        limit = 2 * len(nodes) + 64
        retired = 0
        for t in range(5000):
            node = nodes[t % 8]
            node.key = -t  # every call lowers a key: one stale entry each
            old_before = len(heap._old)
            heap.lowered(node)
            if old_before:
                # a retired heap is only ever drained a few entries at a time
                self.assertLessEqual(old_before - len(heap._old), steps)
            elif heap._old:
                retired += 1
            self.assertLessEqual(heap.num_heap_entries(), limit + limit // steps + 2)
        self.assertGreater(retired, 10)
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), sorted(nodes, key=lambda n: n.key))

    def test_walk_pops_across_a_retired_heap(self):
        nodes = [_Node(i, 1000 + i) for i in range(40)]
        members, heap = self._heap(nodes)
        t = 0
        while not heap._old:  # lower keys until the heap is retired
            t += 1
            nodes[t % 40].key = -t
            heap.lowered(nodes[t % 40])
        for _ in range(3):  # leave valid entries on both sides
            t += 1
            nodes[t % 40].key = -t
            heap.lowered(nodes[t % 40])
        self.assertTrue(heap._old)
        self.assertTrue(heap._heap)
        self._assert_invariants(heap)
        self.assertEqual(_drain(heap), sorted(nodes, key=lambda n: n.key))
        self._assert_invariants(heap)

    def test_check_invariants_reports_divergence(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        stray = _Node(7, 7)
        members.add(stray)  # joined the set without add()
        errors = []
        heap.check_invariants(errors.append, "t", lambda n: True)
        self.assertTrue(any("member without entry" in e for e in errors))
        members.discard(stray)
        members.discard(nodes[2])
        nodes[2].key = -42  # key dropped without lowered(), outside the set
        errors = []
        heap.check_invariants(errors.append, "t", lambda n: True)
        self.assertTrue(any("missed key-drop hook" in e for e in errors))
        errors = []  # deleted without forget()
        heap.check_invariants(errors.append, "t", lambda n: n is not nodes[1])
        self.assertTrue(any("deleted node" in e for e in errors))

    def test_random_walks_match_the_per_walk_rebuild(self):
        # Model check on plain nodes: the persistent heap against a heap that
        # is rebuilt for every walk, under both walk flavours.
        for hide_entrants in (True, False):
            for seed in range(30):
                with self.subTest(hide_entrants=hide_entrants, seed=seed):
                    self._random_walks(random.Random(seed), hide_entrants)

    def _random_walks(self, rng, hide_entrants):
        next_id = [0]

        def new_node():
            next_id[0] += 1
            return _Node(next_id[0], rng.randint(0, 50))

        pool = [new_node() for _ in range(30)]
        members = set(pool[:20])
        heap = _LazyLeafHeap(members, lambda n: n.key, hide_entrants=hide_entrants)
        for n in members:
            heap.add(n)
        for _ in range(40):
            for _ in range(rng.randint(0, 25)):  # between walks
                n = rng.choice(pool)
                r = rng.random()
                if r < 0.35:
                    n.key += rng.randint(0, 9)  # raise: no call
                elif r < 0.55:
                    n.key -= rng.randint(0, 9)
                    heap.lowered(n)
                elif r < 0.75:
                    members.discard(n)  # leave: no call
                elif r < 0.95:
                    if n not in members:
                        members.add(n)
                        heap.add(n)
                else:
                    members.discard(n)
                    heap.forget(n)  # deleted; may come back as a new node
            self._assert_invariants(heap)
            legacy = [(n.key, n) for n in members]
            heapq.heapify(legacy)
            # nodes the per-walk rebuild may yield although they are not in
            # the set (the device walk's explicit pushes)
            pushed = set()
            want, got = [], []
            heap.begin_walk()
            for _ in range(rng.randint(1, 12)):
                x = None
                while legacy:
                    _, cand = heapq.heappop(legacy)
                    if cand in members or cand in pushed:
                        x = cand
                        break
                y = heap.pop_next()
                want.append(x)
                got.append(y)
                if x is None or y is not x:
                    break
                pushed.discard(x)
                if rng.random() < 0.8:
                    members.discard(x)  # evicted; otherwise declined
                r = rng.random()
                if r < 0.3:
                    e = new_node()  # enters the set mid-walk
                    pool.append(e)
                    members.add(e)
                    heap.add(e)
                    if not hide_entrants:
                        heapq.heappush(legacy, (e.key, e))
                    elif rng.random() < 0.7:
                        heapq.heappush(legacy, (e.key, e))
                        heap.promote(e)
                elif r < 0.45 and hide_entrants:
                    e = new_node()  # pushed while outside the set ...
                    pool.append(e)
                    heapq.heappush(legacy, (e.key, e))
                    heap.promote(e)
                    pushed.add(e)
                    if rng.random() < 0.5:  # ... and joins it later in the walk
                        members.add(e)
                        heap.add(e)
                for _ in range(rng.randint(0, 2)):  # a key drops mid-walk
                    n = rng.choice(pool)
                    heap.pin(n)
                    n.key -= rng.randint(1, 9)
                    heap.lowered(n)
                    if not hide_entrants and n not in members and rng.random() < 0.5:
                        members.add(n)  # and it becomes a host leaf again
                        heap.add(n)
                        heapq.heappush(legacy, (n.key, n))
            heap.end_walk()
            self.assertEqual(got, want)
        self._assert_invariants(heap)


# ---------------------------------------------------------------------------
# Order parity against the legacy per-call rebuild
# ---------------------------------------------------------------------------
class TestEvictionOrderParity(CustomTestCase):
    def _assert_parity(self, new: Replay, ref: Replay, label: str):
        self.assertEqual(len(new.victims), len(ref.victims), label)
        for i, (a, b) in enumerate(zip(new.victims, ref.victims)):
            self.assertEqual(a, b, f"{label}: victims differ at evict call {i}")
        self.assertEqual(new.evicted_counts, ref.evicted_counts, label)
        self.assertEqual(new.leaf_paths(), ref.leaf_paths(), label)
        self.assertGreater(sum(len(v) for v in new.victims), 0, label)

    def test_parity_all_policies(self):
        for policy in POLICIES:
            for session in (False, True):
                for seed in range(12):
                    label = f"policy={policy} session={session} seed={seed}"
                    with self.subTest(label):
                        new, ref = replay_pair(policy, seed, session)
                        self._assert_parity(new, ref, label)

    def test_parity_tlru(self):
        # T-LRU keys read the branch high-water depth, which an insert raises
        # on ancestors; a nonzero budget makes the TEL-safe band reachable.
        config = {"threshold": 8, "next_prompt_estimate": 2}
        for session in (False, True):
            for seed in range(6):
                label = f"policy=tlru session={session} seed={seed}"
                with self.subTest(label):
                    new, ref = replay_pair("tlru", seed, session, policy_config=config)
                    self._assert_parity(new, ref, label)

    def test_parity_write_back_queued_parents(self):
        for policy in ("lru", "mru", "priority"):
            for seed in range(4):
                label = f"write_back policy={policy} seed={seed}"
                with self.subTest(label):
                    rng = random.Random(seed)
                    seqs = gen_sequences(rng, 40, 1)
                    orders = []
                    for legacy in (False, True):
                        reset_time_counter()
                        cache = make_cache(policy=policy, legacy=legacy, kv_size=4096)
                        for seq in seqs:
                            _insert(cache, seq)
                        orders.append(queued_walk_order(cache))
                        cache.tree_core.is_write_back = False
                        cache.sanity_check()
                    new, ref = orders
                    self.assertEqual(new, ref, label)
                    # Interior nodes are reached only through queued children.
                    self.assertGreater(
                        len(new[0]), len(cache.tree_core.evictable_device_leaves), label
                    )

    def test_parity_page_size_4(self):
        for policy in ("lru", "lfu", "mru"):
            for seed in range(2):
                label = f"page_size=4 policy={policy} seed={seed}"
                with self.subTest(label):
                    new, ref = replay_pair(policy, seed, False, page_size=4)
                    self._assert_parity(new, ref, label)

    def test_parity_joint_lru_split_restores_the_suffix_time(self):
        # Under joint_lru a split keeps the suffix's older access time, which
        # lowers its key right after the insert touched it. With compaction
        # blocked the eviction is the Full walk.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        orders = []
        for legacy in (False, True):
            reset_time_counter()
            cache, allocator = make_joint_lru_cache(legacy=legacy)
            for tokens in (list(range(1, 11)), [20, 21], list(range(1, 9)) + [50]):
                value = allocator.alloc(len(tokens))
                cache.insert(
                    InsertParams(key=RadixKey(array("q", tokens)), value=value)
                )
            cache.sanity_check()
            core = cache.tree_core
            orig = core.evict_device_leaf
            victims = []

            def recording(node_id, is_write_back, core=core, orig=orig, out=victims):
                out.append(node_path(core, core.node_by_id(node_id)))
                return orig(node_id, is_write_back)

            core.evict_device_leaf = recording
            cache.evict(EvictParams(num_tokens=2))
            cache.sanity_check()
            orders.append(victims)
        new, ref = orders
        self.assertEqual(new, ref)
        # The split-off suffix [9, 10] is older than [20, 21].
        self.assertEqual([path[-1] for path in new], [(9, 10)])


# ---------------------------------------------------------------------------
# Order parity with a host tier (HiCache write_back)
# ---------------------------------------------------------------------------
def write_back_victims(
    policy: str, session: bool, host_tokens: int, seed: int, legacy: bool
) -> list[tuple]:
    """Device and host victims, in order, of the multi-session write_back
    workload of test_write_back_eviction_parity (match, load back, allocate
    with eviction, insert, lock for a step) under ``policy``. A small host pool
    makes device walks evict host leaves and demote queued batches mid-walk.
    With ``session`` every insert registers a session on the leaf and on a
    shorter prefix, and sessions are released now and then."""
    real_params = hicache_write_back_fixtures.CacheInitParams

    def params(**kwargs):
        return real_params(
            eviction_policy=policy,
            enable_session_radix_cache=session,
            component_registry_override=(
                {TREE_TOKEN: LegacyFullComponent} if legacy else None
            ),
            **kwargs,
        )

    reset_time_counter()
    with (
        mock.patch.object(hicache_write_back_fixtures, "CacheInitParams", params),
        mock.patch.object(write_back_parity, "HOST_TOKENS", host_tokens),
        mock.patch.object(write_back_parity, "STEPS", 150),
    ):
        workload = write_back_parity._Workload("python", seed, batched=True, swa=False)
        cache = workload.cache
        core = cache.tree_core
        if legacy:
            forward_host_leaf_adds(cache)
        victims = []
        evict_device_leaf, evict_host_leaf = (
            core.evict_device_leaf,
            core._evict_host_leaf,
        )

        def record_device(node_id, is_write_back):
            victims.append(("device", node_path(core, core.node_by_id(node_id))))
            return evict_device_leaf(node_id, is_write_back)

        def record_host(node, *args, **kwargs):
            victims.append(("host", node_path(core, node)))
            return evict_host_leaf(node, *args, **kwargs)

        core.evict_device_leaf = record_device
        core._evict_host_leaf = record_host
        if session:
            insert = workload._insert

            def insert_and_register(admitted, new_indices, step):
                insert(admitted, new_indices, step)
                key = admitted[1]
                page = hicache_write_back_fixtures.PAGE
                sid = f"s{admitted[0].sid % 4}"
                for cut in (len(key), max(page, len(key) // 2 // page * page)):
                    res = cache.match_prefix(MatchPrefixParams(key=key[:cut]))
                    if res.last_device_node == cache.root_node_handle():
                        continue
                    cache.session_refs.register_session_ref(
                        SimpleNamespace(
                            session_id=sid,
                            session_generation=cache.ensure_session_generation(sid),
                            session=None,
                            last_node=res.last_device_node,
                            origin_input_ids=array("q", key.token_ids[:cut]),
                            output_ids=array("q"),
                            extra_key=None,
                        ),
                        leaf=res.last_device_node,
                    )
                if step % 37 == 0:
                    cache.release_radix_session(f"s{step // 37 % 4}")

            workload._insert = insert_and_register
        workload.run()
    return victims


class TestWriteBackHostTierParity(CustomTestCase):
    def test_parity_write_back_host_tier(self):
        for policy in ("lru", "mru", "filo"):
            for session in (False, True):
                for host_tokens in (32, 128):
                    for seed in range(2):
                        label = (
                            f"write_back policy={policy} session={session} "
                            f"host_tokens={host_tokens} seed={seed}"
                        )
                        with self.subTest(label):
                            new, ref = (
                                write_back_victims(
                                    policy, session, host_tokens, seed, legacy
                                )
                                for legacy in (False, True)
                            )
                            self.assertEqual(new, ref, label)
                            kinds = {kind for kind, _ in new}
                            self.assertEqual(kinds, {"device", "host"}, label)


# ---------------------------------------------------------------------------
# Targeted behaviours on a real cache
# ---------------------------------------------------------------------------
def make_joint_lru_cache(legacy: bool):
    """A Full + SWA tree on the two-ended allocator that keeps the joint LRU,
    with compaction blocked so that eviction takes the Full walk."""
    _, allocator, _ = build_window_chain(
        n_full_slots=48, n_swa_slots=48, full_layer_num=2, swa_layer_num=6
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            sliding_window_size=4,
            tree_slots=tree_slots_of((ComponentType.FULL, ComponentType.SWA), window=4),
            tree_core_backend="python",
            eviction_policy="joint_lru",
            component_registry_override=(
                {TREE_TOKEN: LegacyFullComponent} if legacy else None
            ),
        )
    )
    assert cache.tree_core.joint_lru is not None
    if legacy:
        forward_host_leaf_adds(cache)
    allocator.lazy_compaction = True
    for sub_pool in (allocator.full_attn_allocator, allocator.swa_attn_allocator):
        sub_pool.lazy_compaction = True
        sub_pool.fix_pages("blocked", [sub_pool.min_page_index])
    return cache, allocator


def _insert(cache, tokens):
    v = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    cache.insert(
        InsertParams(key=RadixKey(array("q", tokens)), value=v.to(torch.int64))
    )


def _match_len(cache, tokens) -> int:
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    ).device_prefix_len


def queued_walk_order(cache, rounds: int = 2) -> list[list[tuple]]:
    """Drive Full device walks the way write_back batching does: every yielded
    victim is only queued (``write_back_queued``), so a parent whose device
    children are all queued becomes a walk candidate without being a D-leaf."""
    core = cache.tree_core
    core.is_write_back = True
    full = core.components_by_type[ComponentType.FULL]
    walks = []
    for _ in range(rounds):
        order, queued = [], []
        full.evict_device_start(1 << 30)
        try:
            while True:
                node_id = full.evict_device_next_node(
                    {ComponentType.FULL: 0}, {}, {}, cache_actions=[]
                )
                if node_id is None:
                    break
                node = core.node_by_id(node_id)
                order.append(node_path(core, node))
                node.write_back_queued = True
                queued.append(node)
        finally:
            full.evict_device_end()
        for node in queued:
            node.write_back_queued = False
        walks.append(order)
    return walks


class TestHeapOnRealCache(CustomTestCase):
    def test_mru_touched_leaf_is_evicted_first(self):
        cache = make_cache(policy="mru")
        _insert(cache, [1, 2, 3])
        _insert(cache, [7, 8, 9])
        _match_len(cache, [7, 8, 9])  # most recently used -> first victim under MRU
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(_match_len(cache, [7, 8, 9]), 0)
        self.assertEqual(_match_len(cache, [1, 2, 3]), 3)
        cache.sanity_check()

    def test_lru_touched_leaf_is_protected(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        _insert(cache, [7, 8, 9])
        _match_len(cache, [1, 2, 3])  # refresh -> [7,8,9] is the LRU victim
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(_match_len(cache, [7, 8, 9]), 0)
        self.assertEqual(_match_len(cache, [1, 2, 3]), 3)
        cache.sanity_check()

    def test_session_release_returns_leaf_to_unreferenced_band(self):
        cache = make_cache(policy="lru", enable_session=True)
        _insert(cache, [1, 2, 3, 4])
        _insert(cache, [7, 8, 9])
        res = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
        )
        cache.session_refs.register_session_ref(
            SimpleNamespace(
                session_id="s1",
                session_generation=cache.ensure_session_generation("s1"),
                session=None,
                last_node=res.last_device_node,
                origin_input_ids=array("q", [1, 2, 3, 4]),
                output_ids=array("q"),
                extra_key=None,
            ),
            leaf=res.last_device_node,
        )
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(_match_len(cache, [7, 8, 9]), 0)  # unreferenced goes first
        self.assertEqual(_match_len(cache, [1, 2, 3, 4]), 4)
        cache.release_radix_session("s1")
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(
            _match_len(cache, [1, 2, 3, 4]), 0
        )  # released leaf is evictable again
        cache.sanity_check()

    def test_parent_promotion_within_one_call(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2])
        _insert(cache, [1, 2, 3, 4])
        # One call for 4 tokens must evict the leaf [3,4] and then its parent [1,2].
        res = cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(res.num_tokens_evicted, 4)
        self.assertEqual(_match_len(cache, [1, 2]), 0)
        cache.sanity_check()

    def test_reset_empties_heaps(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        self.assertEqual(len(cache.tree_core.full_device_heap), 1)
        cache.reset()
        self.assertEqual(len(cache.tree_core.full_device_heap), 0)
        self.assertEqual(len(cache.tree_core.full_host_heap), 0)
        cache.sanity_check()

    def test_exception_mid_walk_leaves_heap_consistent(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        _insert(cache, [7, 8, 9])
        core = cache.tree_core
        orig = core.evict_device_leaf

        def boom(node_id, is_write_back):
            raise RuntimeError("injected")

        core.evict_device_leaf = boom
        with self.assertRaises(RuntimeError):
            cache.evict(EvictParams(num_tokens=3))
        core.evict_device_leaf = orig
        self.assertFalse(core.full_device_heap.walking)
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=6))
        self.assertEqual(_match_len(cache, [1, 2, 3]) + _match_len(cache, [7, 8, 9]), 0)
        cache.sanity_check()

    def test_lru_matches_and_lock_churn_push_nothing(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        _insert(cache, [7, 8, 9])
        heap = cache.tree_core.full_device_heap
        before = heap.num_heap_entries()
        for _ in range(2000):
            _match_len(cache, [1, 2, 3])
            res = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [7, 8, 9])))
            )
            lock = cache.inc_lock_ref(res.last_device_node).to_dec_params()
            cache.dec_lock_ref(res.last_device_node, lock)
        self.assertEqual(heap.num_heap_entries(), before)
        cache.sanity_check()

    def test_mru_matches_keep_stale_entries_bounded(self):
        cache = make_cache(policy="mru")
        seqs = [[1000 + i * 3 + j for j in range(3)] for i in range(30)]
        for s in seqs:
            _insert(cache, s)
        heap = cache.tree_core.full_device_heap
        for _ in range(40):
            for s in seqs:
                _match_len(cache, s)
            self.assertLessEqual(heap.num_heap_entries(), 3 * len(heap) + 64)
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=45))
        cache.sanity_check()
        self.assertLessEqual(heap.num_heap_entries(), 3 * len(heap) + 64)


if __name__ == "__main__":
    unittest.main()
