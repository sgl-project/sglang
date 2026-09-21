"""Optional Prometheus export; imported only with serving metrics enabled."""

from prometheus_client import Counter, Gauge

_exporter = None


class CompressionMetrics:
    def __init__(self):
        self.memory = Gauge(
            "sglang_kv_compression_memory_bytes",
            "Compression memory by allocation kind",
            ["kind"],
            multiprocess_mode="livesum",
        )
        self.pages = Counter(
            "sglang_kv_compression_pages",
            "Compression pages by operation",
            ["operation"],
        )
        self.events = Counter(
            "sglang_kv_compression_events",
            "Compression failure or admission events",
            ["operation"],
        )
        self.copies = Counter(
            "sglang_kv_compression_copy_bytes",
            "Compressed L2 copy bytes",
            ["direction"],
        )
        self.seconds = Counter(
            "sglang_kv_compression_seconds",
            "Compression work and restore wall seconds",
            ["operation"],
        )
        self.allocator_work = Counter(
            "sglang_kv_compression_host_allocator_work",
            "Host allocator work counts",
            ["operation"],
        )
        self.allocator_lock_max = Gauge(
            "sglang_kv_compression_host_allocator_max_lock_seconds",
            "Maximum observed Host allocation lock hold time",
            multiprocess_mode="max",
        )
        self.previous = {}

    def _increment(self, collector, label, value):
        key = (id(collector), label)
        before = self.previous.get(key, 0)
        collector.labels(label).inc(max(0, value - before))
        self.previous[key] = value

    def update(self, runtime, adapter, store):
        for key in (
            "metadata_bytes",
            "payload_bytes",
            "reserved_bytes",
            "retired_bytes",
            "arena_bytes",
        ):
            self.memory.labels("l2_" + key).set(store[key])
        for key in (
            "staging_bytes",
            "scratch_bytes",
            "internal_padding_bytes",
            "encoded_bytes",
        ):
            self.memory.labels("l2_" + key).set(store.get(key, 0))
        self.memory.labels("encoded_gpu_resident").set(runtime["resident_bytes"])
        self.memory.labels("encoded_gpu_peak").set(runtime["peak_bytes"])
        for key in (
            "encoded_pages",
            "reused_inflight_pages",
            "reused_host_pages",
            "raw_fallback_pages",
        ):
            self._increment(self.pages, key, runtime[key])
        for key in (
            "backed_up_pages",
            "restored_pages",
            "completed_restore_pages",
            "published_lz4_pages",
            "published_raw_pages",
            "verified_backup_pages",
            "restored_lz4_pages",
            "verified_restored_pages",
        ):
            self._increment(self.pages, key, adapter[key])
        for key in ("backup_failures", "restore_failures", "admission_skips"):
            self._increment(self.events, key, adapter[key])
        for key in ("d2h_bytes", "h2d_bytes"):
            self._increment(self.copies, key, adapter[key])
        for key in (
            "allocation_calls",
            "allocation_failures",
            "allocated_pages",
            "allocation_blocks",
        ):
            self._increment(self.allocator_work, key, store.get(key, 0))
        for key in (
            "allocation_seconds",
            "allocation_lock_wait_seconds",
            "gather_seconds",
            "scatter_seconds",
            "read_stage_wait_seconds",
            "write_stage_wait_seconds",
        ):
            self._increment(self.seconds, key, store.get(key, 0))
        self.allocator_lock_max.set(store.get("allocation_max_lock_seconds", 0))
        self._increment(
            self.seconds, "backup_admission", adapter["backup_admission_seconds"]
        )
        self._increment(self.seconds, "encode", runtime["encode_seconds"])
        self._increment(self.seconds, "restore_wall", adapter["restore_seconds"])
        self._increment(self.seconds, "restore_queue", adapter["restore_queue_seconds"])
        self._increment(
            self.seconds, "restore_execution", adapter["restore_execution_seconds"]
        )


def get_compression_metrics():
    global _exporter
    if _exporter is None:
        _exporter = CompressionMetrics()
    return _exporter
