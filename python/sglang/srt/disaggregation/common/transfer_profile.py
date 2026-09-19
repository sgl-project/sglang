"""Opt-in host timing for a Mooncake DCP transfer chunk.

Submission durations include the synchronous engine call, not just wire time.
Parallel layer calls are summed and can exceed elapsed wall time. No additional
CUDA synchronization is performed; the existing pack function already waits
for its gather stream before returning.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict


class DCPTransferProfile:
    def __init__(self, *, room, worker, source_rank, chunk, enqueued_at):
        self.started = time.perf_counter()
        self.metadata = dict(
            room=room,
            worker=worker,
            source_rank=source_rank,
            page_start=chunk.index_slice.start,
            tokens=chunk.num_kv_tokens,
            last=chunk.is_last_chunk,
            queue_wait_s=(self.started - enqueued_at if enqueued_at else None),
        )
        self.values = defaultdict(int)
        self.lock = threading.Lock()

    def add(self, name, value):
        with self.lock:
            self.values[name] += value

    def call(self, name, func, *args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self.add(name + "_s", time.perf_counter() - start)

    def transfer(self, stage, func, session, blocks):
        self.add(stage + "_calls", 1)
        self.add(stage + "_blocks", len(blocks))
        self.add(stage + "_bytes", sum(b[2] for b in blocks))
        try:
            result = self.call(stage + "_submit", func, session, blocks)
        except Exception:
            self.add(stage + "_failures", 1)
            raise
        if result != 0:
            self.add(stage + "_failures", 1)
        return result

    def batches(self, iterator):
        iterator = iter(iterator)
        while True:
            try:
                batch = self.call("dsa_build", next, iterator)
            except StopIteration:
                return
            yield batch

    def log(self, logger, *, error):
        with self.lock:
            data = dict(self.metadata, **self.values)
        data.update(wall_s=time.perf_counter() - self.started, error=error)
        logger.info("PD_DCP_PROFILE %s", json.dumps(data, sort_keys=True))
