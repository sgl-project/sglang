"""Benchmark-only instrumentation, enabled through the existing plugin loader.

No artificial I/O latency. All A/B/C runs receive the same hooks.
"""

import functools
import json
import os
import time


def event(kind, **data):
    path = os.environ.get("HICACHE_BENCH_TRACE")
    if not path:
        return
    data.update(
        kind=kind,
        label=os.environ.get("HICACHE_BENCH_LABEL"),
        pid=os.getpid(),
        at_ns=time.monotonic_ns(),
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(data, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(fd)


def occupancy(cache):
    pool = cache.cache_controller.mem_pool_host
    return dict(
        host_used_tokens=pool.anchor_entry.host_pool.size - pool.available_size(),
        inflight_tokens=cache.cache_controller.prefetch_tokens_occupied,
    )


def install():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.hicache_storage import HiCacheFile
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if getattr(HiCacheFile, "_benchmark_trace_installed", False):
        return
    HiCacheFile._benchmark_trace_installed = True

    original_get = HiCacheFile.batch_get

    @functools.wraps(original_get)
    def get(self, keys, *args, **kwargs):
        start = time.monotonic_ns()
        result = original_get(self, keys, *args, **kwargs)
        event(
            "read",
            start_ns=start,
            end_ns=time.monotonic_ns(),
            keys=list(keys),
            bytes=sum(x.numel() * x.element_size() for x in result if x is not None),
        )
        return result

    HiCacheFile.batch_get = get

    original_transfer = HybridCacheController._page_transfer

    @functools.wraps(original_transfer)
    def transfer(self, operation):
        start = time.monotonic_ns()
        result = original_transfer(self, operation)
        event(
            "restore_io",
            operation_start_ns=int(operation.start_time * 1e9),
            start_ns=start,
            end_ns=time.monotonic_ns(),
            rid=operation.handle.rid,
            pages=len(operation.hash_value),
        )
        return result

    HybridCacheController._page_transfer = transfer

    original_publish = UnifiedRadixCache._handle_prefetch_result

    @functools.wraps(original_publish)
    def publish(self, operation):
        result = original_publish(self, operation)
        event(
            "publish",
            rid=operation.handle.rid,
            restored_tokens=self.prefetch_loaded_tokens_by_reqid.get(
                operation.handle, 0
            ),
            **occupancy(self),
        )
        return result

    UnifiedRadixCache._handle_prefetch_result = publish

    original_evict = UnifiedRadixCache.evict_host

    @functools.wraps(original_evict)
    def evict(self, *args, **kwargs):
        result = original_evict(self, *args, **kwargs)
        event("evict_host", evicted_tokens=result, **occupancy(self))
        return result

    UnifiedRadixCache.evict_host = evict

    original_load = UnifiedRadixCache.load_back

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        event("h2d_submit", **occupancy(self))
        return original_load(self, *args, **kwargs)

    UnifiedRadixCache.load_back = load

    original_request = Scheduler.handle_generate_request

    @functools.wraps(original_request)
    def request(self, obj):
        event("generation_received", rid=obj.rid, **occupancy(self.tree_cache))
        return original_request(self, obj)

    Scheduler.handle_generate_request = request
