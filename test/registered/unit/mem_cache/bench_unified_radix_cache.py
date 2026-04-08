"""Large-scale benchmark + fuzz correctness tests for UnifiedRadixCache.

Measures insert / match_prefix / evict / lock-unlock / mixed-workload throughput
(tokens/s, ops/s, p50/p99 latency) with randomly generated prefix-sharing
sequences at million-token scale.

Usage (standalone):
    python -m pytest test/registered/unit/mem_cache/bench_unified_radix_cache.py -v -s
    python test/registered/unit/mem_cache/bench_unified_radix_cache.py --num-seqs 5000 --verify
"""

import argparse
import logging
import random
import statistics
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PAGE_SIZE = 1
_HEAD_NUM = 2
_HEAD_DIM = 16
_NUM_LAYERS = 8
_GLOBAL_INTERVAL = 4
_DTYPE = torch.bfloat16
_SWA_WINDOW_SIZE = 128


@contextmanager
def _suppress_logs():
    """Temporarily suppress INFO/DEBUG logs to keep benchmark output clean."""
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(prev_level)


def _full_attention_layer_ids():
    return [i for i in range(_GLOBAL_INTERVAL - 1, _NUM_LAYERS, _GLOBAL_INTERVAL)]


def _mamba_layer_ids():
    full_set = set(_full_attention_layer_ids())
    return [i for i in range(_NUM_LAYERS) if i not in full_set]


def _swa_layer_ids():
    full_set = set(_full_attention_layer_ids())
    return [i for i in range(_NUM_LAYERS) if i not in full_set]


# ===================================================================
# Task 1: Random sequence generator
# ===================================================================
def gen_random_sequences(
    num_seqs: int = 2000,
    chunk_len: int = 256,
    vocab_size: int = 32000,
    seed: int = 42,
) -> list[list[int]]:
    """Generate *num_seqs* token sequences with tree-like prefix sharing.

    Phase 1 (50%): chain growth — each new seq extends a random existing one.
    Phase 2 (50%): fan-out burst — multiple children from the same parent.
    All sequences share a common root prefix of length *chunk_len // 4*.
    """
    rng = random.Random(seed)
    root_prefix = [rng.randint(1, vocab_size) for _ in range(max(1, chunk_len // 4))]
    sequences: list[list[int]] = [root_prefix[:]]

    num_phase1 = num_seqs // 2
    num_phase2 = num_seqs - num_phase1

    # Phase 1: chain growth
    for _ in range(num_phase1):
        parent = rng.choice(sequences)
        suffix_len = rng.randint(1, chunk_len)
        token_id = rng.randint(1, vocab_size)
        child = parent + [token_id] * suffix_len
        sequences.append(child)

    # Phase 2: fan-out burst
    remaining = num_phase2
    while remaining > 0:
        fan = min(rng.randint(2, 10), remaining)
        parent = rng.choice(sequences)
        for _ in range(fan):
            suffix_len = rng.randint(1, chunk_len)
            token_id = rng.randint(1, vocab_size)
            sequences.append(parent + [token_id] * suffix_len)
        remaining -= fan

    rng.shuffle(sequences)
    return sequences


# ===================================================================
# Task 2: Cache factory
# ===================================================================
def create_bench_cache(
    kv_size: int,
    max_num_reqs: int,
    max_context_len: int,
    components: tuple[ComponentType, ...],
    page_size: int = _PAGE_SIZE,
    tree_cls=None,
):
    """Create a UnifiedRadixCache with the given component config and pool sizes.

    Returns (tree, allocator, req_to_token_pool, make_req_fn).
    """
    device = get_device()
    has_mamba = ComponentType.MAMBA in components
    has_swa = ComponentType.SWA in components

    mamba2_cache_params = None
    if has_mamba:
        # Use tiny dimensions — bench tests tree operations, not model compute.
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=256,
                n_groups=1,
                num_heads=2,
                head_dim=16,
                state_size=16,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=_mamba_layer_ids()
            )

    if has_mamba:
        mamba_cache_size = max(max_num_reqs * 2, 200)
        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
    else:
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )

    if has_swa:
        from sglang.srt.mem_cache.swa_memory_pool import (
            SWAKVPool,
            SWATokenToKVPoolAllocator,
        )

        pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size,
            page_size=page_size,
            dtype=_DTYPE,
            head_num=_HEAD_NUM,
            head_dim=_HEAD_DIM,
            swa_attention_layer_ids=_swa_layer_ids(),
            full_attention_layer_ids=_full_attention_layer_ids(),
            enable_kvcache_transpose=False,
            device=device,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size,
            page_size=page_size,
            dtype=_DTYPE,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
    else:
        pool = HybridLinearKVPool(
            size=kv_size,
            dtype=_DTYPE,
            page_size=page_size,
            head_num=_HEAD_NUM,
            head_dim=_HEAD_DIM,
            full_attention_layer_ids=_full_attention_layer_ids(),
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool if has_mamba else None,
        )
        allocator = TokenToKVPoolAllocator(
            size=kv_size,
            dtype=_DTYPE,
            device=device,
            kvcache=pool,
            need_sort=False,
        )

    if tree_cls is None:
        tree_cls = UnifiedRadixCache

    cache_params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        disable=False,
        tree_components=components if tree_cls is UnifiedRadixCache else None,
        sliding_window_size=_SWA_WINDOW_SIZE if has_swa else None,
    )
    tree = tree_cls(params=cache_params)

    _req_counter = [0]

    def make_req():
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        sp = SamplingParams(temperature=0, max_new_tokens=1)
        req = Req(
            rid=_req_counter[0],
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sp,
        )
        _req_counter[0] += 1
        req_to_token_pool.alloc([req])
        return req

    return tree, allocator, req_to_token_pool, make_req


# ===================================================================
# Task 3: Benchmark runner
# ===================================================================
@dataclass
class BenchResult:
    name: str
    num_ops: int
    total_tokens: int
    elapsed_s: float
    latencies_us: list[float]

    @property
    def ops_per_sec(self) -> float:
        return self.num_ops / self.elapsed_s if self.elapsed_s > 0 else 0

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / self.elapsed_s if self.elapsed_s > 0 else 0

    @property
    def p50_us(self) -> float:
        return statistics.median(self.latencies_us) if self.latencies_us else 0

    @property
    def p99_us(self) -> float:
        if not self.latencies_us:
            return 0
        idx = int(len(self.latencies_us) * 0.99)
        return sorted(self.latencies_us)[min(idx, len(self.latencies_us) - 1)]

    def report(self) -> str:
        tok_str = (
            f"{self.tokens_per_sec:>12,.0f} tok/s"
            if self.total_tokens > 0
            else " " * 18
        )
        return (
            f"[bench] {self.name:<20s} | {self.num_ops:>7,d} ops | "
            f"{self.total_tokens:>10,d} tokens | "
            f"{tok_str} | {self.ops_per_sec:>10,.0f} ops/s | "
            f"p50={self.p50_us:>8,.0f}us  p99={self.p99_us:>8,.0f}us"
        )


def bench_api(
    name: str,
    setup_fn: Callable[[], list],
    op_fn: Callable,
    num_ops: int,
    tokens_per_op: int = 0,
    warmup: int = 10,
) -> BenchResult:
    """Time *op_fn(item)* for each item from *setup_fn()*, report throughput."""
    items = setup_fn()
    assert len(items) >= num_ops + warmup, (
        f"setup_fn returned {len(items)} items, need {num_ops + warmup}"
    )

    # Warmup
    for i in range(warmup):
        op_fn(items[i])

    latencies: list[float] = []
    t0 = time.perf_counter()
    for i in range(warmup, warmup + num_ops):
        t_start = time.perf_counter()
        op_fn(items[i])
        latencies.append((time.perf_counter() - t_start) * 1e6)
    elapsed = time.perf_counter() - t0

    total_tokens = tokens_per_op * num_ops if tokens_per_op > 0 else 0
    return BenchResult(
        name=name,
        num_ops=num_ops,
        total_tokens=total_tokens,
        elapsed_s=elapsed,
        latencies_us=latencies,
    )


# ===================================================================
# Task 5: Fuzz correctness helpers
# ===================================================================
def _full_evictable(tree) -> int:
    """Get FULL-component evictable size for both unified and legacy trees."""
    if hasattr(tree, "component_evictable_size_"):
        return tree.component_evictable_size_[ComponentType.FULL]
    return tree.full_evictable_size_


def _full_protected(tree) -> int:
    """Get FULL-component protected size for both unified and legacy trees."""
    if hasattr(tree, "component_protected_size_"):
        return tree.component_protected_size_[ComponentType.FULL]
    return tree.full_protected_size_


def verify_pool_consistency(
    tree,
    allocator,
    label: str = "",
):
    """Assert evictable + protected + free == total for the FULL component."""
    # SWATokenToKVPoolAllocator wraps two sub-allocators; check the full one
    # so the accounting matches FULL-component evictable/protected counters.
    if hasattr(allocator, "full_attn_allocator"):
        pool_available = allocator.full_attn_allocator.available_size()
        pool_total = allocator.full_attn_allocator.size
    else:
        pool_available = allocator.available_size()
        pool_total = allocator.size
    evictable = _full_evictable(tree)
    protected = _full_protected(tree)
    # available + evictable + protected should account for all tokens
    # (minus the root node which has an empty value list)
    total_tracked = pool_available + evictable + protected
    assert total_tracked == pool_total, (
        f"Pool inconsistency [{label}]: "
        f"available({pool_available}) + evictable({evictable}) + protected({protected}) "
        f"= {total_tracked} != pool_size({pool_total})"
    )
    tree.sanity_check()


# ===================================================================
# Task 4: Five benchmark scenarios
# ===================================================================
def _default_components():
    return (ComponentType.FULL, ComponentType.MAMBA)


def bench_insert(
    num_seqs: int = 5000,
    chunk_len: int = 256,
    kv_size: int = 500_000,
    components: tuple[ComponentType, ...] = None,
    verify: bool = False,
    tree_cls=None,
) -> BenchResult:
    """4a. Large-scale insert throughput."""
    if components is None:
        components = _default_components()
    has_mamba = ComponentType.MAMBA in components
    seqs = gen_random_sequences(num_seqs=num_seqs, chunk_len=chunk_len)
    max_seq_len = max(len(s) for s in seqs)
    avg_tokens = sum(len(s) for s in seqs) // len(seqs)

    with _suppress_logs():
        tree, alloc, rtp, make_req = create_bench_cache(
            kv_size=kv_size,
            max_num_reqs=num_seqs + 100,
            max_context_len=max_seq_len + 10,
            components=components,
            tree_cls=tree_cls,
        )

    # Build index list — allocation + insert happen inline in op_fn so
    # the tree can free overlapping prefix KV slots back to the pool.
    items = list(range(len(seqs)))

    def setup_fn():
        return items

    insert_count = [0]

    def op_fn(idx):
        seq = seqs[idx]
        v = alloc.alloc(len(seq))
        if v is None:
            tree.evict(EvictParams(num_tokens=len(seq) * 2, mamba_num=2))
            v = alloc.alloc(len(seq))
        if v is None:
            return  # Pool still full after evict — skip this sequence
        mamba_val = None
        if has_mamba:
            req = make_req()
            mamba_val = req.mamba_pool_idx.unsqueeze(0)
        params = InsertParams(key=RadixKey(seq), value=v, mamba_value=mamba_val)
        tree.insert(params)
        insert_count[0] += 1
        if verify and insert_count[0] % 500 == 0:
            verify_pool_consistency(tree, alloc, f"insert#{insert_count[0]}")

    warmup = min(20, num_seqs // 10)
    num_ops = num_seqs - warmup
    result = bench_api(
        name="insert",
        setup_fn=setup_fn,
        op_fn=op_fn,
        num_ops=num_ops,
        tokens_per_op=avg_tokens,
        warmup=warmup,
    )

    if verify:
        verify_pool_consistency(tree, alloc, "insert-final")

    return result


def bench_match_prefix(
    num_seqs: int = 5000,
    chunk_len: int = 256,
    kv_size: int = 500_000,
    components: tuple[ComponentType, ...] = None,
    verify: bool = False,
    tree_cls=None,
) -> BenchResult:
    """4b. Prefix matching throughput (hit / partial / miss mix)."""
    if components is None:
        components = _default_components()
    has_mamba = ComponentType.MAMBA in components
    seqs = gen_random_sequences(num_seqs=num_seqs, chunk_len=chunk_len)
    max_seq_len = max(len(s) for s in seqs)
    avg_tokens = sum(len(s) for s in seqs) // len(seqs)

    with _suppress_logs():
        tree, alloc, rtp, make_req = create_bench_cache(
            kv_size=kv_size,
            max_num_reqs=num_seqs + 100,
            max_context_len=max_seq_len + 10,
            components=components,
            tree_cls=tree_cls,
        )

    # Populate tree with first half
    populate_count = num_seqs // 2
    for i, seq in enumerate(seqs[:populate_count]):
        v = alloc.alloc(len(seq))
        if v is None:
            tree.evict(EvictParams(num_tokens=len(seq) * 2, mamba_num=2))
            v = alloc.alloc(len(seq))
        if v is None:
            continue
        mamba_val = None
        if has_mamba:
            req = make_req()
            mamba_val = req.mamba_pool_idx.unsqueeze(0)
        tree.insert(InsertParams(key=RadixKey(seq), value=v, mamba_value=mamba_val))

    # Build query mix: 1/3 exact hit, 1/3 partial hit (extended), 1/3 miss
    rng = random.Random(123)
    queries = []
    for seq in seqs:
        roll = rng.random()
        if roll < 0.33:
            # Exact hit — query an inserted sequence
            queries.append(seqs[rng.randint(0, populate_count - 1)])
        elif roll < 0.66:
            # Partial hit — extend an inserted sequence
            base = seqs[rng.randint(0, populate_count - 1)]
            ext = [rng.randint(1, 32000) for _ in range(rng.randint(10, 100))]
            queries.append(base + ext)
        else:
            # Miss — completely new sequence
            queries.append([rng.randint(1, 32000) for _ in range(rng.randint(50, 300))])

    def setup_fn():
        return queries

    match_count = [0]

    def op_fn(query):
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(query)))
        match_count[0] += 1
        if verify and match_count[0] % 1000 == 0:
            # Idempotency: matching same key twice should give same length
            result2 = tree.match_prefix(MatchPrefixParams(key=RadixKey(query)))
            assert len(result.device_indices) == len(result2.device_indices), (
                f"Match not idempotent: {len(result.device_indices)} vs "
                f"{len(result2.device_indices)}"
            )

    warmup = min(20, len(queries) // 10)
    num_ops = min(len(queries) - warmup, num_seqs)
    result = bench_api(
        name="match_prefix",
        setup_fn=setup_fn,
        op_fn=op_fn,
        num_ops=num_ops,
        tokens_per_op=avg_tokens,
        warmup=warmup,
    )
    return result


def bench_evict(
    num_seqs: int = 5000,
    chunk_len: int = 256,
    kv_size: int = 500_000,
    components: tuple[ComponentType, ...] = None,
    verify: bool = False,
    tree_cls=None,
) -> BenchResult:
    """4c. Eviction throughput — fill pool then repeatedly evict batches."""
    if components is None:
        components = _default_components()
    has_mamba = ComponentType.MAMBA in components
    seqs = gen_random_sequences(num_seqs=num_seqs, chunk_len=chunk_len)
    max_seq_len = max(len(s) for s in seqs)

    with _suppress_logs():
        tree, alloc, rtp, make_req = create_bench_cache(
            kv_size=kv_size,
            max_num_reqs=num_seqs + 100,
            max_context_len=max_seq_len + 10,
            components=components,
            tree_cls=tree_cls,
        )

    # Fill tree
    inserted = 0
    for seq in seqs:
        v = alloc.alloc(len(seq))
        if v is None:
            break
        mamba_val = None
        if has_mamba:
            req = make_req()
            mamba_val = req.mamba_pool_idx.unsqueeze(0)
        tree.insert(InsertParams(key=RadixKey(seq), value=v, mamba_value=mamba_val))
        inserted += 1

    evict_batch = max(100, kv_size // 200)
    num_evictions = max(inserted // 5, 100)

    evict_items = [(evict_batch,)] * (num_evictions + 50)

    def setup_fn():
        return evict_items

    evict_count = [0]
    total_evicted_tokens = [0]

    def op_fn(item):
        batch = item[0]
        before_avail = alloc.available_size()
        result = tree.evict(EvictParams(num_tokens=batch, mamba_num=2))
        after_avail = alloc.available_size()
        evicted = result.num_tokens_evicted
        total_evicted_tokens[0] += evicted
        evict_count[0] += 1

        if verify and evict_count[0] % 100 == 0:
            # Evict release consistency
            freed = after_avail - before_avail
            assert freed >= 0, f"Evict freed negative: {freed}"
            verify_pool_consistency(tree, alloc, f"evict#{evict_count[0]}")

    warmup = min(20, num_evictions // 10)
    num_ops = num_evictions - warmup
    result = bench_api(
        name="evict",
        setup_fn=setup_fn,
        op_fn=op_fn,
        num_ops=num_ops,
        tokens_per_op=evict_batch,
        warmup=warmup,
    )
    return result


def bench_lock_unlock(
    num_seqs: int = 5000,
    chunk_len: int = 256,
    kv_size: int = 500_000,
    components: tuple[ComponentType, ...] = None,
    verify: bool = False,
    tree_cls=None,
) -> BenchResult:
    """4d. Lock/unlock throughput — match nodes then cycle lock/unlock."""
    if components is None:
        components = _default_components()
    has_mamba = ComponentType.MAMBA in components
    seqs = gen_random_sequences(num_seqs=num_seqs, chunk_len=chunk_len)
    max_seq_len = max(len(s) for s in seqs)

    with _suppress_logs():
        tree, alloc, rtp, make_req = create_bench_cache(
            kv_size=kv_size,
            max_num_reqs=num_seqs + 100,
            max_context_len=max_seq_len + 10,
            components=components,
            tree_cls=tree_cls,
        )

    # Populate tree
    for seq in seqs[: num_seqs // 2]:
        v = alloc.alloc(len(seq))
        if v is None:
            tree.evict(EvictParams(num_tokens=len(seq) * 2, mamba_num=2))
            v = alloc.alloc(len(seq))
        if v is None:
            continue
        mamba_val = None
        if has_mamba:
            req = make_req()
            mamba_val = req.mamba_pool_idx.unsqueeze(0)
        tree.insert(InsertParams(key=RadixKey(seq), value=v, mamba_value=mamba_val))

    # Collect match nodes
    nodes = []
    for seq in seqs[: num_seqs // 2]:
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        if result.last_device_node != tree.root_node:
            nodes.append(result.last_device_node)

    if not nodes:
        print("[bench] lock_unlock       | SKIPPED (no matchable nodes)")
        return BenchResult("lock_unlock", 0, 0, 0, [])

    # Build lock/unlock pairs
    rng = random.Random(99)
    num_pairs = min(len(nodes) * 2, num_seqs)
    lock_items = [rng.choice(nodes) for _ in range(num_pairs + 50)]

    def setup_fn():
        return lock_items

    lock_count = [0]

    def op_fn(node):
        evictable_before = _full_evictable(tree)
        lock_result = tree.inc_lock_ref(node)
        tree.dec_lock_ref(
            node,
            DecLockRefParams(
                swa_uuid_for_lock=getattr(lock_result, "swa_uuid_for_lock", None)
            ),
        )
        lock_count[0] += 1

        if verify and lock_count[0] % 500 == 0:
            evictable_after = _full_evictable(tree)
            assert evictable_before == evictable_after, (
                f"Lock/unlock asymmetry: evictable {evictable_before} -> {evictable_after}"
            )
            verify_pool_consistency(tree, alloc, f"lock#{lock_count[0]}")

    warmup = min(20, num_pairs // 10)
    num_ops = num_pairs - warmup
    result = bench_api(
        name="lock_unlock",
        setup_fn=setup_fn,
        op_fn=op_fn,
        num_ops=num_ops,
        warmup=warmup,
    )
    return result


def bench_mixed_workload(
    num_seqs: int = 5000,
    chunk_len: int = 256,
    kv_size: int = 500_000,
    components: tuple[ComponentType, ...] = None,
    verify: bool = False,
    tree_cls=None,
) -> BenchResult:
    """4e. Mixed workload — simulates scheduler: insert, match, lock, unlock, evict."""
    if components is None:
        components = _default_components()
    has_mamba = ComponentType.MAMBA in components
    seqs = gen_random_sequences(num_seqs=num_seqs, chunk_len=chunk_len)
    max_seq_len = max(len(s) for s in seqs)
    avg_tokens = sum(len(s) for s in seqs) // len(seqs)

    with _suppress_logs():
        tree, alloc, rtp, make_req = create_bench_cache(
            kv_size=kv_size,
            max_num_reqs=num_seqs + 100,
            max_context_len=max_seq_len + 10,
            components=components,
            tree_cls=tree_cls,
        )

    rng = random.Random(77)

    # Each "op" is one full request lifecycle:
    #   match -> lock -> insert(if new) -> unlock -> maybe evict
    items = list(range(len(seqs)))

    def setup_fn():
        return items

    op_count = [0]

    def op_fn(idx):
        seq = seqs[idx % len(seqs)]
        key = RadixKey(seq)

        # 1. Match prefix
        match_result = tree.match_prefix(MatchPrefixParams(key=key))
        node = match_result.last_device_node
        matched_len = len(match_result.device_indices)

        # 2. Lock
        lock_result = tree.inc_lock_ref(node)

        # 3. Insert remaining (simulate new KV)
        if matched_len < len(seq):
            remaining = len(seq) - matched_len
            v = alloc.alloc(remaining)
            if v is None:
                # Keep node locked so it is not evicted itself
                tree.evict(EvictParams(num_tokens=remaining * 2, mamba_num=2))
                v = alloc.alloc(remaining)
            if v is not None:
                mamba_val = None
                if has_mamba:
                    req = make_req()
                    mamba_val = req.mamba_pool_idx.unsqueeze(0)
                tree.insert(
                    InsertParams(
                        key=key,
                        value=torch.cat([match_result.device_indices, v]),
                        mamba_value=mamba_val,
                        prev_prefix_len=matched_len,
                    )
                )

        # 4. Unlock (exactly once)
        tree.dec_lock_ref(
            node,
            DecLockRefParams(
                swa_uuid_for_lock=getattr(lock_result, "swa_uuid_for_lock", None)
            ),
        )

        # 5. Occasional evict (simulate memory pressure)
        if rng.random() < 0.1:
            tree.evict(EvictParams(num_tokens=100, mamba_num=1))

        op_count[0] += 1
        if verify and op_count[0] % 500 == 0:
            verify_pool_consistency(tree, alloc, f"mixed#{op_count[0]}")

    warmup = min(20, len(seqs) // 10)
    num_ops = len(seqs) - warmup
    result = bench_api(
        name="mixed_workload",
        setup_fn=setup_fn,
        op_fn=op_fn,
        num_ops=num_ops,
        tokens_per_op=avg_tokens,
        warmup=warmup,
    )
    return result


# ===================================================================
# Task 6: Standalone CLI entry
# ===================================================================
ALL_BENCHMARKS = {
    "insert": bench_insert,
    "match": bench_match_prefix,
    "evict": bench_evict,
    "lock": bench_lock_unlock,
    "mixed": bench_mixed_workload,
}


def run_all_benchmarks(
    num_seqs: int = 5000,
    chunk_len: int = 256,
    kv_size: int = 500_000,
    components: tuple[ComponentType, ...] = None,
    verify: bool = False,
    benchmarks: Optional[list[str]] = None,
    tree_cls=None,
):
    if components is None:
        components = _default_components()
    if benchmarks is None or "all" in benchmarks:
        benchmarks = list(ALL_BENCHMARKS.keys())

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", page_size=_PAGE_SIZE)
    )

    impl_name = (tree_cls or UnifiedRadixCache).__name__

    results = []
    for name in benchmarks:
        if name not in ALL_BENCHMARKS:
            print(f"[WARN] Unknown benchmark: {name}, skipping")
            continue
        r = ALL_BENCHMARKS[name](
            num_seqs=num_seqs,
            chunk_len=chunk_len,
            kv_size=kv_size,
            components=components,
            verify=verify,
            tree_cls=tree_cls,
        )
        results.append(r)

    # Print all results at the end, cleanly separated from init logs
    print("=" * 100)
    print(
        f"{impl_name} Benchmark | "
        f"num_seqs={num_seqs}  chunk_len={chunk_len}  kv_size={kv_size}  "
        f"components={[c.value for c in components]}  verify={verify}"
    )
    print("-" * 100)
    for r in results:
        print(r.report())
    print("=" * 100)
    return results


# ===================================================================
# Task 7: pytest wrapper
# ===================================================================
_BENCH_NUM_SEQS = 5000
_BENCH_KV_SIZE = 500_000
_BENCH_CHUNK_LEN = 256


class TestUnifiedRadixCacheBench(unittest.TestCase):
    """Large-scale benchmark + correctness fuzz for UnifiedRadixCache."""

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=_PAGE_SIZE)
        )

    def test_bench_insert(self):
        result = bench_insert(
            num_seqs=_BENCH_NUM_SEQS,
            kv_size=_BENCH_KV_SIZE,
            chunk_len=_BENCH_CHUNK_LEN,
            verify=True,
        )
        self.assertGreater(result.num_ops, 0)
        self.assertGreater(result.ops_per_sec, 0)

    def test_bench_match_prefix(self):
        result = bench_match_prefix(
            num_seqs=_BENCH_NUM_SEQS,
            kv_size=_BENCH_KV_SIZE,
            chunk_len=_BENCH_CHUNK_LEN,
            verify=True,
        )
        self.assertGreater(result.num_ops, 0)
        self.assertGreater(result.ops_per_sec, 0)

    def test_bench_evict(self):
        result = bench_evict(
            num_seqs=_BENCH_NUM_SEQS,
            kv_size=_BENCH_KV_SIZE,
            chunk_len=_BENCH_CHUNK_LEN,
            verify=True,
        )
        self.assertGreater(result.num_ops, 0)

    def test_bench_lock_unlock(self):
        result = bench_lock_unlock(
            num_seqs=_BENCH_NUM_SEQS,
            kv_size=_BENCH_KV_SIZE,
            chunk_len=_BENCH_CHUNK_LEN,
            verify=True,
        )
        self.assertGreater(result.num_ops, 0)

    def test_bench_mixed_workload(self):
        result = bench_mixed_workload(
            num_seqs=_BENCH_NUM_SEQS,
            kv_size=_BENCH_KV_SIZE,
            chunk_len=_BENCH_CHUNK_LEN,
            verify=True,
        )
        self.assertGreater(result.num_ops, 0)
        self.assertGreater(result.ops_per_sec, 0)


# ===================================================================
# CLI
# ===================================================================
# Each config maps to (components, tree_cls).
_TREE_CONFIGS = {
    "full": ((ComponentType.FULL,), None),
    "mamba": ((ComponentType.FULL, ComponentType.MAMBA), None),
    "swa": ((ComponentType.FULL, ComponentType.SWA), None),
    "all": ((ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA), None),
    "legacy-mamba": ((ComponentType.FULL, ComponentType.MAMBA), MambaRadixCache),
    "legacy-swa": ((ComponentType.FULL, ComponentType.SWA), SWARadixCache),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UnifiedRadixCache benchmark & fuzz tester"
    )
    parser.add_argument("--num-seqs", type=int, default=5000)
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument("--kv-size", type=int, default=500_000)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=list(_TREE_CONFIGS.keys()),
        default=["mamba"],
        help="Component configs to benchmark (each produces its own report)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Enable correctness assertions"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmarks to run: insert match evict lock mixed all",
    )
    args = parser.parse_args()

    for comp_name in args.components:
        components, tree_cls = _TREE_CONFIGS[comp_name]
        run_all_benchmarks(
            num_seqs=args.num_seqs,
            chunk_len=args.chunk_len,
            kv_size=args.kv_size,
            components=components,
            verify=args.verify,
            benchmarks=args.benchmarks,
            tree_cls=tree_cls,
        )

