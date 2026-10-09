"""Bounded SHA-256 workers for immutable Host payloads; one caller owns the pool."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, wait

_MIN_PARALLEL_BYTES = 1 << 20


def _digest_group(group):
    return [(index, hashlib.sha256(data).hexdigest()) for index, data in group]


class PayloadHasher:
    def __init__(self, workers: int = 1):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("payload hash workers must be an integer in [1, 8]")
        self.workers = workers
        self.executor = None
        self.closed = False

    def digests(self, buffers: Sequence[memoryview]) -> list[str]:
        """Keep every source immutable until return, including exceptional return."""
        if self.closed:
            raise RuntimeError("payload hasher is closed")
        count = min(self.workers, len(buffers))
        if count <= 1 or sum(data.nbytes for data in buffers) < _MIN_PARALLEL_BYTES:
            return [hashlib.sha256(data).hexdigest() for data in buffers]
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="capture-sha256"
            )
        groups = [[] for _ in range(count)]
        loads = [(0, index) for index in range(count)]
        for index, data in enumerate(buffers):
            size, group = heapq.heappop(loads)
            groups[group].append((index, data))
            heapq.heappush(loads, (size + data.nbytes, group))
        results = [""] * len(buffers)
        futures = []
        try:
            for group in groups:
                futures.append(self.executor.submit(_digest_group, group))
            for future in futures:
                for index, digest in future.result():
                    results[index] = digest
        finally:
            # Submission or one worker can fail while another still reads its
            # arena. No caller may recycle sources until every reader stops.
            wait(futures)
        return results

    def close(self):
        if self.closed:
            return
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        self.closed = True
