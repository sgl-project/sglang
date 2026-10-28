"""Batched write-back eviction makes the per-victim walk's decisions.

A seeded multi-session workload (each prompt extends the session's last one,
host hits load back, the host pool stays full) runs batched and per-victim
through a real UnifiedRadixCache on CPU: both tree cores, Full-only and
FULL+SWA with a window spanning several turns' nodes. After every eviction
call the device and host residency, SWA tombstones, pool sizes and the
evicted, backed-up and loaded-back token counters must match.
"""

import random
import unittest
from array import array
from collections import Counter, deque
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    PAGE,
    hybrid_write_back_cache,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

DEVICE_TOKENS = 64 * PAGE
HOST_TOKENS = 64 * PAGE
# FULL+SWA: a window of three pages over turns that grow by one to three.
WINDOW = 3 * PAGE
SWA_DEVICE_TOKENS = 40 * PAGE
SWA_HOST_TOKENS = 64 * PAGE
SESSIONS = 10
STEPS = 300
MAX_BATCH = 3
# Every few steps the scheduler also asks for a fixed number of tokens.
EVICT_EVERY = 5
SEEDS = (0, 1, 2)


class _Session:
    def __init__(self, sid: int, rng: random.Random, now: int):
        self.sid = sid
        self.prompt_len = PAGE * rng.randint(2, 6)
        self.turns_left = rng.randint(2, 8)
        self.ready_at = now + rng.randint(0, 3)
        self.in_flight = False

    def key(self) -> RadixKey:
        base = self.sid * 100_000
        return RadixKey(array("q", range(base, base + self.prompt_len)))


def _tree_residency(tree_core) -> frozenset:
    """Every node's key path and which tiers hold its component values."""
    nodes = []
    stack = [(tree_core.root_node, ())]
    while stack:
        node, path = stack.pop()
        for child in node.children.values():
            child_path = path + tuple(child.key.token_ids)
            nodes.append(
                (child_path,)
                + tuple(
                    (
                        child.component_data[ct].value is not None,
                        child.component_data[ct].host_value is not None,
                    )
                    for ct in tree_core.component_types
                )
            )
            stack.append((child, child_path))
    return frozenset(nodes)


class _CacheMetrics:
    """The cache counters behind sglang:evicted_tokens_total,
    sglang:hicache_backup_tokens_total and sglang:load_back_tokens_total."""

    def __init__(self):
        self.totals = Counter()

    def increment_eviction_num_tokens(self, num_tokens):
        self.totals["evicted"] += num_tokens

    def increment_backup_num_tokens(self, num_tokens, pool):
        self.totals["backup", pool] += num_tokens

    def increment_load_back_num_tokens(self, num_tokens, pool):
        self.totals["load_back", pool] += num_tokens

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Workload:
    """Seeded sessions driven the way the scheduler drives the cache: match,
    load the host hit back, allocate the extend (evicting for it), insert, and
    hold the request's lock for a step."""

    def __init__(self, tree_core_backend: str, seed: int, batched: bool, swa: bool):
        self.swa = swa
        if swa:
            self.cache, self.allocator, self.host_pool, self.swa_host_pool = (
                hybrid_write_back_cache(
                    tree_core_backend,
                    device_tokens=DEVICE_TOKENS,
                    swa_device_tokens=SWA_DEVICE_TOKENS,
                    host_tokens=HOST_TOKENS,
                    swa_host_tokens=SWA_HOST_TOKENS,
                    sliding_window=WINDOW,
                    enable_kv_cache_events=True,
                )
            )
        else:
            self.cache, self.allocator, self.host_pool = write_back_cache(
                tree_core_backend,
                device_tokens=DEVICE_TOKENS,
                host_tokens=HOST_TOKENS,
                enable_kv_cache_events=True,
            )
        self.python_core = tree_core_backend == "python"
        self.metrics = self.cache.metrics_collector = _CacheMetrics()
        if not batched:
            # Each victim is copied and demoted before the walk advances.
            self.cache._write_back_batch_tokens = lambda *args: 0
        self.rng = random.Random(seed)
        self.next_sid = 0
        self.sessions = [self._new_session(0) for _ in range(SESSIONS)]
        self.locks = deque()  # (release step, lock, session)
        self.inserted_keys = {}  # sid -> last inserted key
        self.resident = {"GPU": Counter(), "CPU_PINNED": Counter()}
        self.snapshots = []
        self.copy_waits = 0
        self._instrument()

    def _new_session(self, now: int) -> _Session:
        session = _Session(self.next_sid, self.rng, now)
        self.next_sid += 1
        return session

    def _instrument(self):
        cache = self.cache
        for name in ("evict", "evict_for_alloc"):
            real = getattr(cache, name)

            def evict(params, _real=real, _name=name, **kwargs):
                result = _real(params, **kwargs)
                cache.sanity_check()
                self.snapshots.append((_name, params.num_tokens, self._state()))
                return result

            setattr(cache, name, evict)
        writing_check = cache.writing_check

        def count_copy_waits(write_back=False, finish_count=None):
            if write_back and cache.ongoing_write_through:
                self.copy_waits += 1
            return writing_check(write_back=write_back, finish_count=finish_count)

        cache.writing_check = count_copy_waits

    @property
    def swa_allocator(self) -> SWATokenToKVPoolAllocator:
        assert isinstance(self.allocator, SWATokenToKVPoolAllocator)
        return self.allocator

    def _full_available(self) -> int:
        if self.swa:
            return self.swa_allocator.full_available_size()
        return self.allocator.available_size()

    def _state(self) -> dict:
        cache = self.cache
        for event in cache.take_events():
            if isinstance(event, (BlockStored, BlockRemoved)):
                assert event.medium is not None
                blocks = self.resident[event.medium]
                for block in event.block_hashes:
                    blocks[block] += 1 if isinstance(event, BlockStored) else -1
                    if blocks[block] == 0:
                        del blocks[block]
        state: dict[str, object] = {
            medium: frozenset(blocks) for medium, blocks in self.resident.items()
        }
        state["sizes"] = (
            cache.full_evictable_size(),
            self._full_available(),
            self.host_pool.available_size(),
        )
        if self.swa:
            tree_core = cache.tree_core
            state["swa tombstones"] = tuple(
                tuple(tree_core.swa_tombstone_ranges(key, 0, len(key)))
                for _, key in sorted(self.inserted_keys.items())
            )
            state["swa sizes"] = (
                cache.swa_evictable_size(),
                self.swa_allocator.swa_available_size(),
                self.swa_host_pool.available_size(),
            )
        if self.python_core:
            state["tree"] = _tree_residency(cache.tree_core)
        return state

    def _admit(self, session: _Session, batch_extend: int):
        cache = self.cache
        key = session.key()
        match = cache.match_prefix(MatchPrefixParams(key=key))
        device_len, host_len = match.device_prefix_len, match.host_hit_length
        swa_host_len = match.swa_host_hit_length if self.swa else 0
        # Lock the hit first: it is not evictable room for the request itself.
        admit_lock = cache.lock(match.last_device_node)
        extend = session.prompt_len - device_len + batch_extend
        if extend > self._full_available() + cache.full_evictable_size() or (
            self.swa
            and extend + WINDOW + PAGE
            > self.swa_allocator.swa_available_size() + cache.swa_evictable_size()
        ):
            cache.unlock(admit_lock)
            return None
        last_node, loaded = match.last_device_node, 0
        if host_len > 0 or swa_host_len > 0:
            best = match.best_match_node
            host_lock = cache.inc_host_lock_ref(best).to_dec_params()
            best_lock = cache.lock(best)
            req = SimpleNamespace(
                rid=f"s{session.sid}",
                last_node=last_node,
                swa_host_hit_length=swa_host_len,
                mamba_host_hit_length=0,
                kv=SimpleNamespace(mamba_pool_idx=None),
            )
            try:
                load = cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=best,
                        host_hit_length=host_len,
                        req=req,
                    )
                )
                assert load is not None  # only buffer mode returns None
                loaded, last_node = load
            finally:
                cache.unlock(best_lock)
                cache.dec_host_lock_ref(best, host_lock)
            # A failed Full load keeps the device match; an SWA-only load may
            # still expose device Full that lacked its window.
            if loaded == 0 and host_len > 0:
                last_node = match.last_device_node
        lock = cache.lock(last_node)
        cache.unlock(admit_lock)
        return session, key, device_len + loaded, last_node, lock

    def _swa_evicted_seqlen(self, admitted, new_indices) -> int:
        """Free the request's own SWA slots below its window's frontier, as
        checkpoint_into_tree does, and return the insert's SWA evicted length."""
        session, _, prefix_len, _, _ = admitted
        pre_len = session.prompt_len - 1
        if pre_len - WINDOW < 1:
            return 0
        insert_stop = (pre_len + 1) // PAGE * PAGE
        frontier = min(pre_len - max(WINDOW, PAGE), insert_stop - WINDOW)
        frontier = max(prefix_len, frontier) // PAGE * PAGE
        if frontier > prefix_len:
            own = new_indices[: frontier - prefix_len].to(torch.int64)
            self.swa_allocator.free_swa_segment(own, start_pos=0)
            return frontier
        return prefix_len

    def _insert(self, admitted, new_indices, step: int):
        cache = self.cache
        session, key, prefix_len, last_node, lock = admitted
        prefix = cache.path_device_indices(last_node).to(torch.int64)
        assert len(prefix) == prefix_len, (len(prefix), prefix_len)
        params = InsertParams(
            key=key,
            value=torch.cat([prefix, new_indices.to(torch.int64)]),
            prev_prefix_len=prefix_len,
        )
        if self.swa:
            params.set_evicted_seqlen(
                ComponentType.SWA, self._swa_evicted_seqlen(admitted, new_indices)
            )
        result = cache.insert(params)
        self.inserted_keys[session.sid] = key
        new_lock = cache.lock(result.last_device_node)
        cache.unlock(lock)
        self.locks.append((step + 1, new_lock, session))

    def _end_turn(self, session: _Session, step: int):
        session.in_flight = False
        session.turns_left -= 1
        if session.turns_left == 0:
            self.sessions[self.sessions.index(session)] = self._new_session(step)
            return
        session.prompt_len += PAGE * self.rng.randint(1, 3)
        session.ready_at = step + self.rng.randint(1, 6)

    def _alloc(self, batch, extends):
        total = sum(extends)
        if total == 0:
            return torch.empty((0,), dtype=torch.int64)
        self.allocator.evict_to_free_tokens(self.cache, total)
        if self.swa:
            prefix_lens = torch.tensor([admitted[2] for admitted in batch])
            seq_lens = torch.tensor([admitted[0].prompt_len for admitted in batch])
            indices = self.allocator.alloc_extend(
                prefix_lens,
                prefix_lens,
                seq_lens,
                seq_lens,
                torch.full((len(batch),), -1, dtype=torch.int64),
                total,
            )
        else:
            indices = self.allocator.alloc(total)
        assert indices is not None
        return indices

    def run(self):
        cache = self.cache
        for step in range(STEPS):
            cache.check_hicache_events()
            while self.locks and self.locks[0][0] <= step:
                _, lock, session = self.locks.popleft()
                cache.unlock(lock)
                self._end_turn(session, step)
            ready = sorted(
                (s for s in self.sessions if s.ready_at <= step and not s.in_flight),
                key=lambda s: (s.ready_at, s.sid),
            )
            batch, extends = [], []
            for session in ready[:MAX_BATCH]:
                admitted = self._admit(session, sum(extends))
                if admitted is None:
                    break
                session.in_flight = True
                batch.append(admitted)
                extends.append(session.prompt_len - admitted[2])
            indices = self._alloc(batch, extends)
            cache.ready_to_load_host_cache()
            cache.check_hicache_events()
            for admitted, chunk in zip(batch, torch.split(indices, extends)):
                self._insert(admitted, chunk, step)
            if step % EVICT_EVERY == 0:
                cache.evict(EvictParams(num_tokens=PAGE * self.rng.randint(1, 8)))
        cache.sanity_check()
        return self.snapshots, self.copy_waits, self.metrics.totals


class _WriteBackParitySuite:
    tree_core_backend: str

    def _assert_parity(self, swa: bool):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                per_victim, per_victim_waits, per_victim_metrics = _Workload(
                    self.tree_core_backend, seed, batched=False, swa=swa
                ).run()
                batched, batched_waits, batched_metrics = _Workload(
                    self.tree_core_backend, seed, batched=True, swa=swa
                ).run()
                for call, (got, want) in enumerate(zip(batched, per_victim)):
                    kind, num_tokens, state = got
                    self.assertEqual(
                        (kind, num_tokens),
                        want[:2],
                        f"eviction call {call} asked for something else",
                    )
                    for name, value in state.items():
                        if value != want[2][name]:
                            self.fail(
                                f"{name} differ after eviction call {call} ({kind} {num_tokens})"
                            )
                self.assertEqual(len(batched), len(per_victim))
                self.assertLess(batched_waits, per_victim_waits)
                self.assertEqual(batched_metrics, per_victim_metrics)
                self.assertEqual(
                    {
                        key if isinstance(key, str) else key[0]
                        for key in batched_metrics
                    },
                    {"evicted", "backup", "load_back"},
                )

    def test_batched_eviction_makes_the_per_victim_decisions(self):
        self._assert_parity(swa=False)

    def test_batched_swa_eviction_makes_the_per_victim_decisions(self):
        self._assert_parity(swa=True)


class TestWriteBackParityRustCore(_WriteBackParitySuite, CustomTestCase):
    tree_core_backend = "rust"


class TestWriteBackParityPythonCore(_WriteBackParitySuite, CustomTestCase):
    tree_core_backend = "python"


if __name__ == "__main__":
    unittest.main()
