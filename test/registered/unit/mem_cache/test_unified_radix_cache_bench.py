"""Large-scale benchmark + fuzz correctness tests for UnifiedRadixCache.

Usage (standalone):
    bench: python3 test/registered/unit/mem_cache/test_unified_radix_cache_bench.py --num-seqs 5000 --verify --components mamba legacy-mamba swa legacy-swa
    CI Test: python -m pytest test/registered/unit/mem_cache/test_unified_radix_cache_bench.py -v -s
"""

import argparse
import gc
import logging
import random
import statistics
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

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

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")

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

_BENCH_NUM_SEQS = 5000
_BENCH_KV_SIZE = 500_000
_BENCH_CHUNK_LEN = 256

_DEFAULT_COMPONENTS = (ComponentType.FULL, ComponentType.MAMBA)


@contextmanager
def _suppress_logs():
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(prev)


def _full_attention_layer_ids():
    return list(range(_GLOBAL_INTERVAL - 1, _NUM_LAYERS, _GLOBAL_INTERVAL))


def _non_full_layer_ids():
    full = set(_full_attention_layer_ids())
    return [i for i in range(_NUM_LAYERS) if i not in full]


# ===================================================================
# Sequence generator
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
    """
    rng = random.Random(seed)
    root_prefix = [rng.randint(1, vocab_size) for _ in range(max(1, chunk_len // 4))]
    sequences: list[list[int]] = [root_prefix[:]]

    # Phase 1: chain growth
    for _ in range(num_seqs // 2):
        parent = rng.choice(sequences)
        sequences.append(
            parent + [rng.randint(1, vocab_size)] * rng.randint(1, chunk_len)
        )

    # Phase 2: fan-out burst
    remaining = num_seqs - num_seqs // 2
    while remaining > 0:
        fan = min(rng.randint(2, 10), remaining)
        parent = rng.choice(sequences)
        for _ in range(fan):
            sequences.append(
                parent + [rng.randint(1, vocab_size)] * rng.randint(1, chunk_len)
            )
        remaining -= fan

    rng.shuffle(sequences)
    return sequences


# ===================================================================
# Cache factory
# ===================================================================
def create_bench_cache(
    kv_size,
    max_num_reqs,
    max_context_len,
    components,
    page_size=_PAGE_SIZE,
    tree_cls=None,
):
    """Create cache.  Returns (tree, allocator, req_to_token_pool, make_req)."""
    device = get_device()
    has_mamba = ComponentType.MAMBA in components
    has_swa = ComponentType.SWA in components

    mamba2_cache_params = None
    if has_mamba:
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
                shape=shape, layers=_non_full_layer_ids()
            )

    # --- req_to_token pool ---
    if has_mamba:
        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=max(max_num_reqs * 2, 200),
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=_non_full_layer_ids(),
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

    # --- KV pool + allocator ---
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
            swa_attention_layer_ids=_non_full_layer_ids(),
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

    # --- tree ---
    if tree_cls is None:
        tree_cls = UnifiedRadixCache
    tree = tree_cls(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            tree_components=components if tree_cls is UnifiedRadixCache else None,
            sliding_window_size=_SWA_WINDOW_SIZE if has_swa else None,
        )
    )

    _rid = [0]

    def make_req():
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid=_rid[0],
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        _rid[0] += 1
        req_to_token_pool.alloc([req])
        return req

    return tree, allocator, req_to_token_pool, make_req


# ===================================================================
# Shared bench environment + helpers
# ===================================================================
@dataclass
class _Env:
    tree: object
    alloc: object
    rtp: object
    make_req: Callable
    seqs: list
    has_mamba: bool
    avg_tokens: int


def _make_env(num_seqs, chunk_len, kv_size, components, tree_cls=None):
    """Create sequences + cache, return shared _Env."""
    if components is None:
        components = _DEFAULT_COMPONENTS
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
    return _Env(
        tree, alloc, rtp, make_req, seqs, ComponentType.MAMBA in components, avg_tokens
    )


def _alloc_with_evict(env, n):
    """Alloc *n* tokens, evicting if necessary.  Returns tensor or None."""
    v = env.alloc.alloc(n)
    if v is None:
        env.tree.evict(EvictParams(num_tokens=n * 2, mamba_num=2))
        v = env.alloc.alloc(n)
    return v


def _insert_seq(env, seq):
    """Insert one sequence (alloc + evict-fallback).  Returns True on success."""
    v = _alloc_with_evict(env, len(seq))
    if v is None:
        return False
    mamba_val = None
    if env.has_mamba:
        req = env.make_req()
        mamba_val = req.mamba_pool_idx.unsqueeze(0)
    env.tree.insert(InsertParams(key=RadixKey(seq), value=v, mamba_value=mamba_val))
    return True


def _populate(env, count):
    """Insert first *count* sequences (with evict-fallback)."""
    for seq in env.seqs[:count]:
        _insert_seq(env, seq)


def _fill_no_evict(env):
    """Insert sequences until pool exhausted (no eviction).  Returns count."""
    inserted = 0
    for seq in env.seqs:
        v = env.alloc.alloc(len(seq))
        if v is None:
            break
        mamba_val = None
        if env.has_mamba:
            req = env.make_req()
            mamba_val = req.mamba_pool_idx.unsqueeze(0)
        env.tree.insert(InsertParams(key=RadixKey(seq), value=v, mamba_value=mamba_val))
        inserted += 1
    return inserted


# ===================================================================
# Benchmark result + runner
# ===================================================================
@dataclass
class BenchResult:
    name: str
    num_ops: int
    total_tokens: int
    elapsed_s: float
    latencies_us: list[float]

    @property
    def ops_per_sec(self):
        return self.num_ops / self.elapsed_s if self.elapsed_s > 0 else 0

    @property
    def tokens_per_sec(self):
        return self.total_tokens / self.elapsed_s if self.elapsed_s > 0 else 0

    @property
    def p50_us(self):
        return statistics.median(self.latencies_us) if self.latencies_us else 0

    @property
    def p99_us(self):
        if not self.latencies_us:
            return 0
        idx = int(len(self.latencies_us) * 0.99)
        return sorted(self.latencies_us)[min(idx, len(self.latencies_us) - 1)]

    def report(self):
        tok = (
            f"{self.tokens_per_sec:>12,.0f} tok/s"
            if self.total_tokens > 0
            else f"{'N/A':>12s} tok/s"
        )
        return (
            f"  {self.name:<18s} | {tok} | {self.ops_per_sec:>10,.0f} ops/s | "
            f"p50={self.p50_us:>8,.0f}us  p99={self.p99_us:>8,.0f}us"
        )


def bench_api(
    name, setup_fn, op_fn, num_ops, tokens_per_op=0, warmup=10, verify_fn=None
):
    """Time *op_fn(item)* for each item from *setup_fn()*.

    *verify_fn*, if provided, runs during warmup and once after timing
    (excluded from latency measurement).
    """
    items = setup_fn()
    assert (
        len(items) >= num_ops + warmup
    ), f"need {num_ops + warmup} items, got {len(items)}"

    for i in range(warmup):
        op_fn(items[i])
        if verify_fn:
            verify_fn(items[i])

    gc.collect()
    gc_was = gc.isenabled()
    gc.disable()

    latencies: list[float] = []
    t0 = time.perf_counter()
    for i in range(warmup, warmup + num_ops):
        ts = time.perf_counter()
        op_fn(items[i])
        latencies.append((time.perf_counter() - ts) * 1e6)
    elapsed = time.perf_counter() - t0

    if gc_was:
        gc.enable()
    if verify_fn:
        verify_fn(items[warmup + num_ops - 1])

    return BenchResult(
        name,
        num_ops,
        tokens_per_op * num_ops if tokens_per_op > 0 else 0,
        elapsed,
        latencies,
    )


# ===================================================================
# Five benchmark scenarios
# ===================================================================
def bench_insert(
    num_seqs=5000,
    chunk_len=256,
    kv_size=500_000,
    components=None,
    verify=False,
    tree_cls=None,
):
    """Insert throughput (alloc + evict-fallback + insert)."""
    env = _make_env(num_seqs, chunk_len, kv_size, components, tree_cls)
    warmup = min(20, num_seqs // 10)

    return bench_api(
        "insert",
        lambda: list(range(len(env.seqs))),
        lambda idx: _insert_seq(env, env.seqs[idx]),
        num_seqs - warmup,
        env.avg_tokens,
        warmup,
        (lambda _: env.tree.sanity_check()) if verify else None,
    )


def bench_match_prefix(
    num_seqs=5000,
    chunk_len=256,
    kv_size=500_000,
    components=None,
    verify=False,
    tree_cls=None,
):
    """Prefix matching throughput (hit / partial / miss mix)."""
    env = _make_env(num_seqs, chunk_len, kv_size, components, tree_cls)
    _populate(env, num_seqs // 2)

    rng = random.Random(123)
    pop = num_seqs // 2
    queries: list[list[int]] = []
    for _ in env.seqs:
        roll = rng.random()
        if roll < 0.33:
            queries.append(env.seqs[rng.randint(0, pop - 1)])
        elif roll < 0.66:
            base = env.seqs[rng.randint(0, pop - 1)]
            queries.append(base + [rng.randint(1, 32000)] * rng.randint(10, 100))
        else:
            queries.append([rng.randint(1, 32000)] * rng.randint(50, 300))

    def verify_fn(q):
        r1 = env.tree.match_prefix(MatchPrefixParams(key=RadixKey(q)))
        r2 = env.tree.match_prefix(MatchPrefixParams(key=RadixKey(q)))
        assert len(r1.device_indices) == len(r2.device_indices), "match not idempotent"

    warmup = min(20, len(queries) // 10)
    return bench_api(
        "match_prefix",
        lambda: queries,
        lambda q: env.tree.match_prefix(MatchPrefixParams(key=RadixKey(q))),
        min(len(queries) - warmup, num_seqs),
        env.avg_tokens,
        warmup,
        verify_fn if verify else None,
    )


def bench_evict(
    num_seqs=5000,
    chunk_len=256,
    kv_size=500_000,
    components=None,
    verify=False,
    tree_cls=None,
):
    """Eviction throughput — fill pool then repeatedly evict batches."""
    env = _make_env(num_seqs, chunk_len, kv_size, components, tree_cls)
    inserted = _fill_no_evict(env)

    evict_batch = max(100, kv_size // 200)
    num_evictions = max(inserted // 5, 100)
    items = [(evict_batch,)] * (num_evictions + 50)
    warmup = min(20, num_evictions // 10)

    return bench_api(
        "evict",
        lambda: items,
        lambda item: env.tree.evict(EvictParams(num_tokens=item[0], mamba_num=2)),
        num_evictions - warmup,
        evict_batch,
        warmup,
        (lambda _: env.tree.sanity_check()) if verify else None,
    )


def bench_lock_unlock(
    num_seqs=5000,
    chunk_len=256,
    kv_size=500_000,
    components=None,
    verify=False,
    tree_cls=None,
):
    """Lock/unlock throughput — match nodes then cycle lock/unlock."""
    env = _make_env(num_seqs, chunk_len, kv_size, components, tree_cls)
    _populate(env, num_seqs // 2)

    nodes = []
    for seq in env.seqs[: num_seqs // 2]:
        r = env.tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        if r.last_device_node != env.tree.root_node:
            nodes.append(r.last_device_node)
    if not nodes:
        return BenchResult("lock_unlock", 0, 0, 0, [])

    rng = random.Random(99)
    num_pairs = min(len(nodes) * 2, num_seqs)
    items = [rng.choice(nodes) for _ in range(num_pairs + 50)]

    def op_fn(node):
        lr = env.tree.inc_lock_ref(node)
        env.tree.dec_lock_ref(
            node,
            DecLockRefParams(swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)),
        )

    warmup = min(20, num_pairs // 10)
    return bench_api(
        "lock_unlock",
        lambda: items,
        op_fn,
        num_pairs - warmup,
        0,
        warmup,
        (lambda _: env.tree.sanity_check()) if verify else None,
    )


def bench_cache_finished(
    num_seqs=5000,
    chunk_len=256,
    kv_size=500_000,
    components=None,
    verify=False,
    tree_cls=None,
):
    """cache_finished_req throughput — full request lifecycle.

    Simulates: match_prefix → inc_lock_ref → alloc → fill req_to_token → cache_finished_req.
    """
    env = _make_env(num_seqs, chunk_len, kv_size, components, tree_cls)

    # Pre-build Req objects with token IDs filled into req_to_token
    req_items: list = []
    for seq in env.seqs:
        key = RadixKey(seq)
        mr = env.tree.match_prefix(MatchPrefixParams(key=key))
        matched_len = len(mr.device_indices)
        node = mr.last_device_node
        lr = env.tree.inc_lock_ref(node)

        remaining = len(seq) - matched_len
        if remaining > 0:
            v = _alloc_with_evict(env, remaining)
            if v is None:
                env.tree.dec_lock_ref(
                    node,
                    DecLockRefParams(
                        swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)
                    ),
                )
                continue
            kv_indices = torch.cat([mr.device_indices, v])
        else:
            kv_indices = mr.device_indices

        req = env.make_req()
        req.origin_input_ids = list(seq)
        req.output_ids = []
        req.fill_ids = list(seq)
        req.last_node = node
        req.cache_protected_len = matched_len
        req.kv_committed_len = len(seq)
        req.kv_committed_freed = False
        if hasattr(lr, "swa_uuid_for_lock"):
            req.swa_uuid_for_lock = lr.swa_uuid_for_lock
        env.rtp.req_to_token[req.req_pool_idx, : len(kv_indices)] = kv_indices
        req_items.append(req)

    if not req_items:
        return BenchResult("cache_finished", 0, 0, 0, [])

    warmup = min(20, len(req_items) // 10)
    return bench_api(
        "cache_finished",
        lambda: req_items,
        lambda req: env.tree.cache_finished_req(req, is_insert=True),
        len(req_items) - warmup,
        env.avg_tokens,
        warmup,
        # Pool math doesn't hold here (many reqs still hold allocated tokens).
        (lambda _: env.tree.sanity_check()) if verify else None,
    )


# ===================================================================
# Runner
# ===================================================================
ALL_BENCHMARKS = {
    "insert": bench_insert,
    "match": bench_match_prefix,
    "evict": bench_evict,
    "lock": bench_lock_unlock,
    "cache_finished": bench_cache_finished,
}


def run_all_benchmarks(
    num_seqs=5000,
    chunk_len=256,
    kv_size=500_000,
    components=None,
    verify=False,
    benchmarks=None,
    tree_cls=None,
):
    if components is None:
        components = _DEFAULT_COMPONENTS
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
        results.append(
            ALL_BENCHMARKS[name](
                num_seqs=num_seqs,
                chunk_len=chunk_len,
                kv_size=kv_size,
                components=components,
                verify=verify,
                tree_cls=tree_cls,
            )
        )

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
# pytest wrapper
# ===================================================================
class TestUnifiedRadixCacheBench(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=_PAGE_SIZE)
        )

    def test_bench_insert(self):
        r = bench_insert(_BENCH_NUM_SEQS, _BENCH_CHUNK_LEN, _BENCH_KV_SIZE, verify=True)
        self.assertGreater(r.num_ops, 0)
        self.assertGreater(r.ops_per_sec, 0)

    def test_bench_match_prefix(self):
        r = bench_match_prefix(
            _BENCH_NUM_SEQS, _BENCH_CHUNK_LEN, _BENCH_KV_SIZE, verify=True
        )
        self.assertGreater(r.num_ops, 0)
        self.assertGreater(r.ops_per_sec, 0)

    def test_bench_evict(self):
        r = bench_evict(_BENCH_NUM_SEQS, _BENCH_CHUNK_LEN, _BENCH_KV_SIZE, verify=True)
        self.assertGreater(r.num_ops, 0)

    def test_bench_lock_unlock(self):
        r = bench_lock_unlock(
            _BENCH_NUM_SEQS, _BENCH_CHUNK_LEN, _BENCH_KV_SIZE, verify=True
        )
        self.assertGreater(r.num_ops, 0)

    def test_bench_cache_finished(self):
        r = bench_cache_finished(
            _BENCH_NUM_SEQS, _BENCH_CHUNK_LEN, _BENCH_KV_SIZE, verify=True
        )
        self.assertGreater(r.num_ops, 0)
        self.assertGreater(r.ops_per_sec, 0)


# ===================================================================
# CLI
# ===================================================================
_TREE_CONFIGS = {
    "full": ((ComponentType.FULL,), None),
    "mamba": ((ComponentType.FULL, ComponentType.MAMBA), None),
    "swa": ((ComponentType.FULL, ComponentType.SWA), None),
    "all": ((ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA), None),
    "legacy-mamba": ((ComponentType.FULL, ComponentType.MAMBA), MambaRadixCache),
    "legacy-swa": ((ComponentType.FULL, ComponentType.SWA), SWARadixCache),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UnifiedRadixCache benchmark")
    parser.add_argument("--num-seqs", type=int, default=5000)
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument("--kv-size", type=int, default=500_000)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=list(_TREE_CONFIGS.keys()),
        default=["mamba", "legacy-mamba"],
        help="Component configs to benchmark",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Enable correctness assertions"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="insert match evict lock cache_finished all",
    )
    args, _ = parser.parse_known_args()

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
