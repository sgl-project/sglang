"""Compressed host byte I/O. No cache tree, request or transport ownership."""

import contextlib
import time
from collections import defaultdict

import torch
from sglang.srt.kv_compression.store import materialize_pages
from sglang.srt.kv_compression.types import BufferDrainError
from sglang.srt.kv_compression.verification import page_digests


class CompressedHostIO:
    def __init__(self, runtime, host_pool):
        self.runtime, self.host_pool = runtime, host_pool
        self.stats = defaultdict(float)
        self.backup_progress = {}
        self.quarantined = []

    def _source_checksums(self, indices, ready, stream):
        charge = 2 * len(indices) * self.runtime.layout.page_bytes
        self.runtime._charge(charge)
        raw = None
        try:
            with (
                torch.cuda.stream(stream)
                if stream is not None
                else contextlib.nullcontext()
            ):
                if stream is not None:
                    stream.wait_event(ready)
                raw = self.runtime.layout.pack_pages(indices)
                return page_digests(raw)
        finally:
            try:
                if stream is not None:
                    stream.synchronize()
            except Exception as exc:
                self.runtime.quarantine(exc, (raw, indices))
                raise BufferDrainError("L2 source verification did not drain") from exc
            with self.runtime._lock:
                self.runtime._live_bytes -= charge

    def _write(self, handles, refs, indices, ready):
        # The tree source pin covers the whole node; GPU payload leases cover
        # one window only. Sharing is opportunistic, never unbounded retention.
        stream = (
            torch.cuda.Stream(device=self.runtime.device)
            if self.runtime.device.type == "cuda"
            else None
        )
        leases = []
        total = 0
        try:
            for start in range(0, len(refs), self.runtime.batch_pages):
                end = start + self.runtime.batch_pages
                leases = self.runtime.acquire_pages(
                    refs[start:end], indices[start:end], ready, priority=2
                )
                self.stats["backup_window_pages"] = max(
                    self.stats["backup_window_pages"], len(leases)
                )
                wait_started = time.perf_counter()
                pages = [lease.future.result() for lease in leases]
                self.stats["backup_wait_seconds"] += time.perf_counter() - wait_started
                checksums = [None] * len(pages)
                if self.host_pool.verify:
                    # Snapshot the protected source independently of encoded
                    # output, including when an existing object was reused.
                    verify_started = time.perf_counter()
                    checksums = self._source_checksums(
                        indices[start:end], ready, stream
                    )
                    self.stats["backup_verify_seconds"] += (
                        time.perf_counter() - verify_started
                    )
                write = self.host_pool.prepare_write(
                    handles[start:end], [page.nbytes for page in pages]
                )
                try:
                    # Source results/resources are ready before either channel.
                    # Host reuse, if any, always takes read before write.
                    with (
                        materialize_pages(pages) as contiguous,
                        self.host_pool.stage(read=False) as stage,
                    ):
                        copy_started = time.perf_counter()
                        copy_events = (
                            [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            if stream is not None
                            else []
                        )
                        if copy_events:
                            copy_events[0].record(stream)
                        try:
                            with (
                                torch.cuda.stream(stream)
                                if stream is not None
                                else contextlib.nullcontext()
                            ):
                                for i, page in enumerate(contiguous):
                                    stage[i, : page.nbytes].copy_(
                                        page.data, non_blocking=stream is not None
                                    )
                        finally:
                            if stream is not None:
                                try:
                                    copy_events[1].record(stream)
                                    stream.synchronize()
                                    self.stats["backup_d2h_gpu_ms"] += copy_events[
                                        0
                                    ].elapsed_time(copy_events[1])
                                except Exception as exc:
                                    raise BufferDrainError(
                                        "L2 D2H did not drain"
                                    ) from exc
                        self.stats["backup_d2h_seconds"] += (
                            time.perf_counter() - copy_started
                        )
                        write.scatter(stage)
                        publish_started = time.perf_counter()
                        write.publish(refs[start:end], pages, checksums)
                        self.stats["backup_publish_seconds"] += (
                            time.perf_counter() - publish_started
                        )
                    for page, checksum in zip(pages, checksums):
                        self.stats[f"published_{page.encoding}_pages"] += 1
                        self.stats["verified_backup_pages"] += int(checksum is not None)
                        total += page.nbytes
                except BufferDrainError:
                    write.quarantine()
                    raise
                finally:
                    write.close()
                self.backup_progress["completed_pages"] = min(end, len(refs))
                for lease in leases:
                    lease.close()
                leases.clear()
                pages.clear()
                page = None
            return total
        finally:
            failure = None
            for lease in leases:
                if lease.future is None:
                    continue
                try:
                    lease.future.result()
                except BufferDrainError as exc:
                    failure = exc
                except Exception:
                    pass
            try:
                if stream is not None:
                    stream.synchronize()
            except Exception as exc:
                failure = BufferDrainError(f"L2 D2H did not drain: {exc}")
            if failure is not None:
                self.quarantined.append((leases, handles, refs))
                raise failure
            for lease in leases:
                lease.close()
