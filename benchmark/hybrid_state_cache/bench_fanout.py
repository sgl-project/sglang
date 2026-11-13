"""CPU stress benchmark of the hybrid-state cache switches under a wide shared
prefix: how the radix cache's operations scale with the number of session
branches below one prefix.

The tree, on the unified radix cache with full attention and Mamba states
(the CPU test kit, page size 1): a 256-token shared prefix holding a state and
N session branches of 32 tokens below it; half the branches end in a state and
half hold KV only, and a quarter of them split into a stateless interior node
and a second 16-token sub-branch. Each operation starts from a fresh tree and
is timed alone; untimed steps between calls restore the tree, so N stays
fixed while it repeats. Each pool holds the tree plus a fixed headroom, so
the allocator's free lists do not grow with N.

  match           match_prefix of a branch plus one token
  insert          a session's next turn: its branch + 16 new tokens, half of
                  them with a state
  cache_finished  the same turn as a request: match, lock, row, then what
                  releasing a finished request does (checkpoint, free, unlock)
  evict_tokens    evict(num_tokens=16), the evicted branches restored after
  evict_units     evict(mamba_num=1); under gds the first call takes all KV
                  that no state serves, so only branches with a state return
                  (the slots it frees are held, keeping the free list short)
  gds_score_*,    GreedyDual-Size scoring and pushing the shared prefix's unit
  gds_push_*      (N branches below it) and a leaf's unit

--heap-run instead runs a long gds workload with few evictions (90% hits, 9%
new turns, 1% state evictions) and records the heap sizes.

Examples:

    python benchmark/hybrid_state_cache/bench_fanout.py --config gds \\
        --out /tmp/gds.json
    python benchmark/hybrid_state_cache/bench_fanout.py --config units \\
        --n 100 1000 10000 --out /tmp/units.json
    python benchmark/hybrid_state_cache/bench_fanout.py --heap-run \\
        --n 1000 --ops 40000 --out /tmp/heap.json

Run it under taskset on one idle core; compare trees by running the same
command against each (PYTHONPATH=<tree>/python); --config off also runs on a
tree without the switches.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from array import array
from dataclasses import fields
from unittest.mock import patch

SHARED = list(range(1, 257))


class FanoutTree:
    def __init__(self, config: str, n: int, reps: int, seed: int = 0):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.test.kits.unified_radix_cache.config import CacheConfig
        from sglang.test.kits.unified_radix_cache.fixtures import build_fixture

        self.MAMBA = ComponentType.MAMBA
        self.n = n
        self.rng = random.Random(seed)
        self.next_token = 100_000
        self.rid = 0
        # [tokens, holds a state] per branch.
        self.branches: list[list] = []
        for _ in range(n):
            u = self.fresh(32)
            self.branches.append([SHARED + u, self.rng.random() < 0.5])
            if self.rng.random() < 0.25:
                self.branches.append(
                    [SHARED + u[:16] + self.fresh(16), self.rng.random() < 0.5]
                )
        tokens = len(SHARED) + self.next_token - 100_000
        states = 1 + sum(state for _, state in self.branches)
        # What the tree holds plus a fixed headroom: the CPU allocator's free
        # copies its free list, so the free list must not grow with n.
        kw = dict(
            components=(ComponentType.FULL, ComponentType.MAMBA),
            kv_size=tokens + 16384 + 32 * reps,
            mamba_cache_size=states + 1024 + 8 * reps,
            max_num_reqs=states + 1024 + 8 * reps,
            max_context_len=1024,
            num_layers=4,
            full_attention_layer_ids=(3,),
            head_num=1,
            head_dim=8,
            mamba_intermediate_size=16,
            mamba_num_heads=1,
            mamba_head_dim=16,
            mamba_state_size=16,
        )
        if config == "units":
            kw["enable_mamba_unit_eviction"] = True
        elif config == "gds":
            kw["eviction_policy"] = "gds"
            kw["eviction_policy_config"] = {"prefill_cost": [1.0, 0.0]}
        missing = set(kw) - {f.name for f in fields(CacheConfig)}
        assert not missing, f"this tree cannot run --config {config}: {missing}"
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            self.cache, self.allocator, self.pool = build_fixture(CacheConfig(**kw))
        self.core = self.cache.tree_core
        self.kv_capacity = self.allocator.available_size()
        # Free slots taken out of play (hold), so that the free list keeps the
        # headroom's length.
        self.held: list = []
        self.gds = getattr(self.core, "gds", None)
        if self.gds is not None:
            self.gds.state_tokens = 10.0
        self.insert(SHARED, state=False)
        for tokens, state in self.branches:
            self.insert(tokens, state=state)
        self.insert(SHARED, state=True)
        self.shared = self.find(SHARED)
        assert self.shared is not None and len(self.shared.children) == n

    # ---- helpers ----

    def fresh(self, k: int) -> list[int]:
        start, self.next_token = self.next_token, self.next_token + k
        return list(range(start, start + k))

    def new_req(self, tokens=()):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        self.rid += 1
        return Req(
            rid=f"fanout-{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )

    def insert_params(self, tokens: list[int], state: bool):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.static_kv_storage import acquire_rows

        value = self.allocator.alloc(len(tokens))
        assert value is not None, "the KV pool is exhausted"
        params = InsertParams(key=RadixKey(array("q", tokens)), value=value)
        if state:
            req = self.new_req()
            acquire_rows(self.pool, self.pool.state, [req])
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        else:
            params.allow_missing_mamba = True
        return params

    def insert(self, tokens: list[int], state: bool) -> None:
        # The cache frees the slots of the tokens it already holds.
        self.cache.insert(self.insert_params(tokens, state))

    def find(self, tokens: list[int]):
        """The node ending at ``tokens``, without touching the tree."""
        from sglang.srt.mem_cache.radix_cache import RadixKey

        key = RadixKey(array("q", tokens))
        node, depth = self.core.root_node, 0
        while depth < len(tokens):
            child = node.children.get(key.child_key_at(depth, 1))
            if child is None:
                return None
            k = len(child.key)
            if list(child.key.raw_token_ids()) != tokens[depth : depth + k]:
                return None
            node, depth = child, depth + k
        return node

    def restore(self, stateful_only: bool = False) -> None:
        """Insert again every branch that lost its leaf or its state."""
        for tokens, state in self.branches:
            if stateful_only and not state:
                continue
            node = self.find(tokens)
            if node is None or (
                state and node.component_data[self.MAMBA].value is None
            ):
                self.insert(tokens, state=state)

    def hold(self, n: int) -> None:
        if n > 0:
            self.held.append(self.allocator.alloc(n))

    def check(self, full_fanout: bool) -> None:
        self.cache.sanity_check()
        used = len(self.core.all_values_flatten()) + sum(map(len, self.held))
        assert self.allocator.available_size() + used == self.kv_capacity, "KV leak"
        if full_fanout:
            assert len(self.shared.children) == self.n, "the fan-out drifted"

    # ---- timed operations: each returns nanoseconds ----

    def op_match(self) -> int:
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        tokens = self.rng.choice(self.branches)[0] + [7]
        params = MatchPrefixParams(key=RadixKey(array("q", tokens)))
        t = time.perf_counter_ns()
        self.cache.match_prefix(params)
        return time.perf_counter_ns() - t

    def op_insert(self) -> int:
        tokens = self.rng.choice(self.branches)[0] + self.fresh(16)
        params = self.insert_params(tokens, state=self.rng.random() < 0.5)
        t = time.perf_counter_ns()
        self.cache.insert(params)
        return time.perf_counter_ns() - t

    def op_cache_finished(self) -> int:
        from sglang.srt.mem_cache.static_kv_storage import acquire_rows
        from sglang.test.mem_cache_utils import finish_req

        tokens = self.rng.choice(self.branches)[0] + self.fresh(16)
        req = self.new_req(tokens)
        req.init_next_round_input(self.cache)
        req.lock = self.cache.lock(req.last_node)
        prefix = self.cache.prefix_device_indices(req) if req.prefix_len else None
        acquire_rows(self.pool, self.pool.state, [req])
        row = req.kv.req_pool_idx
        if prefix is not None:
            self.pool.write((row, slice(0, req.prefix_len)), prefix)
        new = self.allocator.alloc(len(tokens) - req.prefix_len)
        self.pool.write((row, slice(req.prefix_len, len(tokens))), new)
        req.output_ids = array("q")
        req.extra_key = None
        req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = req.prefix_len
        req.kv.mamba_last_track_seqlen = len(tokens)
        t = time.perf_counter_ns()
        finish_req(self.cache, req, len(tokens))
        return time.perf_counter_ns() - t

    def op_evict(self, **kwargs) -> int:
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        params = EvictParams(**kwargs)
        t = time.perf_counter_ns()
        self.cache.evict(params)
        return time.perf_counter_ns() - t


def timed(fn, *args) -> int:
    t = time.perf_counter_ns()
    fn(*args)
    return time.perf_counter_ns() - t


def summarize(samples: list[int]) -> dict:
    samples = sorted(samples)
    return {
        "median_us": statistics.median(samples) / 1000,
        "p90_us": samples[int(0.9 * (len(samples) - 1))] / 1000,
        "mean_us": statistics.fmean(samples) / 1000,
        "reps": len(samples),
    }


def run_phase(fn, reps: int, after=None) -> dict:
    gc.collect()
    gc.disable()
    try:
        samples = []
        for _ in range(reps):
            samples.append(fn())
            if after is not None:
                after()
        return summarize(samples)
    finally:
        gc.enable()


def bench(config: str, n: int, reps: int) -> dict:
    res = {}

    def fresh() -> FanoutTree:
        return FanoutTree(config, n, reps)

    for name in ("match", "insert", "cache_finished"):
        tree = fresh()
        res[name] = run_phase(getattr(tree, f"op_{name}"), reps)
        tree.check(full_fanout=True)
    gds = config == "gds"
    if gds:
        tree = fresh()
        leaves = [
            x
            for x in tree.core._node_arena.values()
            if not x.children and x.component_data[tree.MAMBA].value is not None
        ][:50]
        for name, fn, pick in (
            ("gds_score_shared", tree.gds.score, lambda: tree.shared),
            ("gds_score_leaf", tree.gds.score, lambda: tree.rng.choice(leaves)),
            ("gds_push_shared", tree.gds.push, lambda: tree.shared),
            ("gds_push_leaf", tree.gds.push, lambda: tree.rng.choice(leaves)),
        ):
            res[name] = run_phase(lambda: timed(fn, pick()), reps)
        tree.check(full_fanout=True)
    evict_reps = max(20, reps // 3)
    tree = fresh()
    res["evict_tokens"] = run_phase(
        lambda: tree.op_evict(num_tokens=16), evict_reps, after=tree.restore
    )
    tree.check(full_fanout=True)
    tree = fresh()
    if gds:
        # The first state request takes all KV that no state serves; hold the
        # slots it freed, as the allocator's free copies its free list.
        free = tree.allocator.available_size()
        tree.op_evict(mamba_num=1)
        tree.restore(stateful_only=True)
        tree.hold(tree.allocator.available_size() - free)
    res["evict_units"] = run_phase(
        lambda: tree.op_evict(mamba_num=1),
        evict_reps,
        after=lambda: tree.restore(stateful_only=gds),
    )
    tree.check(full_fanout=not gds)
    res["evictable_leaves"] = len(tree.core.evictable_device_leaves)
    return res


def heap_run(n: int, ops: int, seed: int = 0) -> list[dict]:
    tree = FanoutTree("gds", n, reps=ops // 50 + 50, seed=seed)
    units = getattr(tree.core, "state_units", None)
    rng = random.Random(seed + 1)
    trace = []
    gc.disable()
    try:
        for i in range(ops):
            r = rng.random()
            if r < 0.90:
                tree.op_match()
            elif r < 0.99:
                tree.op_insert()
            else:
                tree.op_evict(mamba_num=1)
                tree.restore(stateful_only=True)
            if (i + 1) % max(1, ops // 20) == 0:
                trace.append(
                    {
                        "ops": i + 1,
                        "heap": len(tree.gds._heap),
                        "units": len(tree.gds._version),
                        "dead_queue": len(getattr(units, "_dead", ())),
                        "evictable_leaves": len(tree.core.evictable_device_leaves),
                    }
                )
    finally:
        gc.enable()
    return trace


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", choices=("off", "units", "gds"), default="gds")
    ap.add_argument("--n", type=int, nargs="+", default=[100, 1000, 10000])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--heap-run", action="store_true")
    ap.add_argument("--ops", type=int, default=40000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.heap_run:
        out = {n: heap_run(n, args.ops) for n in args.n}
        for n, trace in out.items():
            for row in trace:
                print(n, row)
    else:
        out = {}
        for n in args.n:
            out[n] = bench(args.config, n, args.reps)
        ops = [k for k in out[args.n[0]] if k != "evictable_leaves"]
        print(f"--config {args.config}: median us per call")
        print(f"{'operation':18s}" + "".join(f"{'N=' + str(n):>12s}" for n in args.n))
        for op in ops:
            row = "".join(f"{out[n][op]['median_us']:12.1f}" for n in args.n)
            print(f"{op:18s}{row}")
    with open(args.out, "w") as f:
        json.dump({"config": args.config, "results": out}, f, indent=1)


if __name__ == "__main__":
    main()
