#!/usr/bin/env python3

import argparse
import json
import time

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStore,
    _FailedGetCache,
)


class DelayedStaleBackend:
    def __init__(self, get_delay_seconds: float):
        self.get_delay_seconds = get_delay_seconds
        self.exists_calls = 0
        self.get_calls = 0

    def batch_is_exist(self, keys):
        self.exists_calls += 1
        return [1] * len(keys)

    def batch_get_into(self, keys, buffer_ptrs, buffer_sizes):
        self.get_calls += 1
        time.sleep(self.get_delay_seconds)
        return [-5] * len(keys)


def run_case(iterations: int, get_delay_seconds: float, ttl_seconds: float):
    backend = DelayedStaleBackend(get_delay_seconds)
    store = MooncakeStore.__new__(MooncakeStore)
    store.store = backend
    store.failed_get_cache = (
        _FailedGetCache(ttl_seconds, max_entries=1024) if ttl_seconds > 0 else None
    )

    started = time.perf_counter()
    for _ in range(iterations):
        if store._batch_exist(["stale-page"])[0] == 1:
            store._get_batch_zero_copy_impl(["stale-page"], [0x1000], [4096])
    elapsed = time.perf_counter() - started

    return (
        {
            "iterations": iterations,
            "get_delay_ms": get_delay_seconds * 1000,
            "ttl_seconds": ttl_seconds,
            "elapsed_ms": elapsed * 1000,
            "remote_exists_calls": backend.exists_calls,
            "remote_get_calls": backend.get_calls,
        },
        store,
        backend,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--get-delay-ms", type=float, default=10.0)
    parser.add_argument("--ttl-seconds", type=float, default=1.0)
    args = parser.parse_args()

    delay_seconds = args.get_delay_ms / 1000
    baseline, _, _ = run_case(args.iterations, delay_seconds, 0)
    enabled, store, backend = run_case(args.iterations, delay_seconds, args.ttl_seconds)

    time.sleep(args.ttl_seconds + 0.05)
    exists_before_retry = backend.exists_calls
    retry_visible = store._batch_exist(["stale-page"])[0] == 1

    result = {
        "baseline": baseline,
        "enabled": enabled,
        "elapsed_reduction_pct": 100
        * (baseline["elapsed_ms"] - enabled["elapsed_ms"])
        / baseline["elapsed_ms"],
        "get_suppression_pct": 100
        * (baseline["remote_get_calls"] - enabled["remote_get_calls"])
        / baseline["remote_get_calls"],
        "ttl_retry": {
            "remote_exists_calls_added": backend.exists_calls - exists_before_retry,
            "remote_hit_visible_after_expiry": retry_visible,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
