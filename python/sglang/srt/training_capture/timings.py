"""Bounded wall-time accounting for background snapshot work, not GPU kernels."""

import math
import threading
from contextlib import contextmanager
from time import perf_counter


class CaptureTimings:
    STAGES = (
        "queue_wait",
        "copy_wait",
        "snapshot_build",
        "validation",
        "catalog_register",
        "store_payload",
        "catalog_written",
        "journal_save",
        "catalog_seal",
        "store_manifest",
        "catalog_publish",
        "journal_complete",
        "recovery_read",
    )

    def __init__(self, *, clock=perf_counter):
        self.clock = clock
        self.lock = threading.Lock()
        self.values = {
            stage: {"calls": 0, "errors": 0, "seconds": 0.0, "max_seconds": 0.0}
            for stage in self.STAGES
        }

    def observe(self, stage, seconds, *, failed=False):
        if stage not in self.values:
            raise ValueError("unknown capture timing stage")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid capture stage duration")
        with self.lock:
            value = self.values[stage]
            value["calls"] += 1
            value["errors"] += int(failed)
            value["seconds"] += seconds
            value["max_seconds"] = max(value["max_seconds"], seconds)

    @contextmanager
    def measure(self, stage):
        if stage not in self.values:
            raise ValueError("unknown capture timing stage")
        started = self.clock()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            self.observe(stage, max(0.0, self.clock() - started), failed=not succeeded)

    def call(self, stage, callback, *args, **kwargs):
        with self.measure(stage):
            return callback(*args, **kwargs)

    def stats(self):
        with self.lock:
            return {stage: dict(value) for stage, value in self.values.items()}
