"""Bounded, page-based execution shared by P/D and HiCache.

This module knows neither requests nor radix trees. Consumers acquire leases;
only HiCache can retain a result after the last transient consumer releases it.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import heapq
import itertools
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import torch

from sglang.srt.kv_compression.store import HostEncodedPage, materialize_pages
from sglang.srt.kv_compression.types import (
    BufferDrainError,
    CompressionCapacityError,
    EncodedPage,
    KVVerificationError,
    Lease,
    new_page_refs,
)
from sglang.srt.kv_compression.verification import page_digests

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    future: concurrent.futures.Future
    users: int = 0
    resident_bytes: int = 0
    job: object = None
    submitted: bool = False


class WorkspaceReservation:
    """Explicit ownership through the final consumer, not just the producer Future."""

    def __init__(self, runtime, size):
        self.runtime, self.size = runtime, size
        self.closed = False
        self.quarantined = False

    def release(self):
        with self.runtime._lock:
            if self.quarantined:
                raise BufferDrainError("Workspace is quarantined")
            if not self.closed:
                self.closed = True
                self.runtime._live_bytes -= self.size
                self.runtime._workspace_bytes -= self.size

    def quarantine(self, error, owners=()):
        # Do not release the charge or the tensors while access is uncertain.
        self.quarantined = True
        self.runtime.quarantine(error, (self, owners))


class DecodedPages:
    """Borrow raw/staging tensors only while this owner remains open.

    close() must name the consumer stream when it differs from the runtime
    stream. Callers must drop borrowed tensor views before closing the owner.
    Backend-private CUDA allocations are separate from this explicit budget.
    """

    def __init__(self, runtime, raw, reservation):
        self.runtime, self.raw, self.reservation = runtime, raw, reservation
        self.staging = None
        self.conversion_stream = None

    def to_staging_order(self):
        if self.reservation.closed:
            raise RuntimeError("Decoded pages have been released")
        if self.staging is None:
            if self.raw.is_cuda:
                self.conversion_stream = torch.cuda.current_stream(self.raw.device)
            self.staging = self.runtime.layout.to_staging_order(self.raw)
        return self.staging

    def close(self, consumer_stream=None):
        if self.reservation.closed:
            return
        try:
            if self.reservation.quarantined:
                raise BufferDrainError("Decoded pages are quarantined")
            self.runtime.drain()
            if (
                self.conversion_stream is not None
                and self.conversion_stream is not consumer_stream
            ):
                self.conversion_stream.synchronize()
            if consumer_stream is not None:
                consumer_stream.synchronize()
        except Exception as exc:
            self.reservation.quarantine(exc, (self, consumer_stream))
            raise BufferDrainError("Decoded consumer did not drain") from exc
        self.raw = self.staging = None
        self.reservation.release()


class KVCompressionRuntime:
    batch_pages = 64

    def __init__(
        self,
        layout,
        mode,
        budget_bytes,
        backend_factory=None,
        *,
        force=False,
        verify=False,
    ):
        if mode not in ("passthrough", "lz4"):
            raise ValueError("Unsupported compression execution mode")
        if force and mode != "lz4":
            raise ValueError("Forced compression requires LZ4")
        self.force = force
        self.verify = verify
        self.trace_reuse = os.environ.get(
            "SGLANG_KV_COMPRESSION_TRACE_REUSE", "0"
        ).lower() in ("1", "true", "yes", "y")
        self.layout, self.mode = layout, mode
        self.device = layout.device
        self.budget_bytes = budget_bytes
        self.provider = None
        self._factory = backend_factory
        self._lock = threading.RLock()
        self._entries = {}
        self._jobs = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._live_bytes = 0
        self._workspace_bytes = 0
        self._workspace_waiters = 0
        self._closed = False
        self._running = 0
        self._active_job = None
        self._quarantine = []
        self.stats = {
            "encoded_pages": 0,
            "reused_inflight_pages": 0,
            "reused_host_pages": 0,
            "raw_fallback_pages": 0,
            "encode_seconds": 0.0,
            "peak_bytes": 0,
            "workspace_rejections": 0,
            "workspace_wait_seconds": 0.0,
            "attempted_raw_bytes": 0,
            "attempted_encoded_bytes": 0,
            "attempted_capacity_bytes": 0,
            "nonshrinking_pages": 0,
            "insufficient_saving_pages": 0,
            "queue_seconds": 0.0,
            "pack_gpu_ms": 0.0,
            "encode_gpu_ms": 0.0,
            "ready_wait_gpu_ms": 0.0,
            "pack_host_seconds": 0.0,
            "encode_host_seconds": 0.0,
            "restore_copy_gpu_ms": 0.0,
            "decompress_gpu_ms": 0.0,
            "decompress_host_seconds": 0.0,
            "restore_copy_host_seconds": 0.0,
            "verified_restore_pages": 0,
            "restore_verification_failures": 0,
            "writeback_gpu_ms": 0.0,
            "queue_restore_seconds": 0.0,
            "queue_send_seconds": 0.0,
            "queue_backup_seconds": 0.0,
        }
        self._initialized = concurrent.futures.Future()
        self._worker = threading.Thread(
            target=self._run, name="kv-compression", daemon=True
        )
        self._worker.start()
        # Initialization is before request admission, not scheduler polling.
        self._initialized.result()

    def _run(self):
        try:
            self.stream = (
                torch.cuda.Stream(device=self.device)
                if self.device.type == "cuda"
                else None
            )
            if self.mode == "lz4":
                if self._factory is None:
                    from sglang.srt.kv_compression.backend import NvcompLZ4Backend

                    self._factory = NvcompLZ4Backend
                self.backend = self._factory(self.device, self.stream)
                sample = torch.empty(
                    self.layout.page_bytes, dtype=torch.uint8, device=self.device
                )
                self.output_bound = self.backend.max_output_bytes(sample)
                del sample
            else:
                self.backend = None
                self.output_bound = self.layout.page_bytes
            self._initialized.set_result(None)
        except BaseException as exc:
            self._initialized.set_exception(exc)
            return
        while True:
            priority, job_id, future, fn, queued_at = self._jobs.get()
            if fn is None:
                return
            if not future.set_running_or_notify_cancel():
                fn = future = None
                continue
            result = None
            try:
                with self._lock:
                    self._running += 1
                    wait = time.monotonic() - queued_at
                    self.stats["queue_seconds"] += wait
                    category = {0: "restore", 1: "send", 2: "backup"}.get(priority)
                    if category:
                        self.stats[f"queue_{category}_seconds"] += wait
                    self._active_job = {
                        "job_id": job_id,
                        "priority": priority,
                        "queued_at": queued_at,
                        "started_at": time.monotonic(),
                    }
                    quarantined = bool(self._quarantine)
                if quarantined:
                    raise BufferDrainError("Compression runtime is quarantined")
                with self.stream_context():
                    result = fn()
                future.set_result(result)
            except BaseException as exc:
                if isinstance(exc, BufferDrainError):
                    with self._lock:
                        # Keep the failed task's borrowed buffers alive and
                        # reject queued work as well as new submissions.
                        if not self._quarantine:
                            self._quarantine.append(fn)
                future.set_exception(exc)
            finally:
                with self._lock:
                    self._running -= 1
                    self._active_job = None
                # A sleeping worker must not retain the previous GPU payload
                # through its completed Future or closure.
                fn = future = result = None

    def stream_context(self):
        return (
            torch.cuda.stream(self.stream)
            if self.stream is not None
            else contextlib.nullcontext()
        )

    def drain(self):
        if self.stream is not None:
            try:
                self.stream.synchronize()
            except Exception as exc:
                raise BufferDrainError(
                    "Shared compression stream did not drain"
                ) from exc

    def submit(self, fn, priority=1):
        with self._lock:
            if self._closed or self._quarantine:
                raise RuntimeError("Compression runtime is closed or quarantined")
            if self._jobs.qsize() >= 256:
                raise CompressionCapacityError("Compression task queue is full")
            future = concurrent.futures.Future()
            self._jobs.put(
                [priority, next(self._sequence), future, fn, time.monotonic()]
            )
            return future

    def _charge(self, size):
        with self._lock:
            if size < 0 or size > self.budget_bytes:
                raise CompressionCapacityError(
                    "Single operation exceeds compression workspace budget"
                )
            if self._live_bytes + size > self.budget_bytes:
                self.stats["workspace_rejections"] += 1
                raise CompressionCapacityError(
                    "Compression GPU workspace budget exhausted"
                )
            self._live_bytes += size
            self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self._live_bytes)

    def reserve_workspace(self, size, *, wait=False, cancelled=None):
        if size < 0 or size > self.budget_bytes:
            raise CompressionCapacityError(
                "Single operation exceeds compression workspace budget"
            )
        if wait and threading.current_thread() is self._worker:
            raise RuntimeError(
                "The sole compression worker must not wait for workspace"
            )
        started = time.monotonic()
        with self._lock:
            self._workspace_waiters += 1
        try:
            while True:
                with self._lock:
                    if cancelled is not None and cancelled():
                        raise concurrent.futures.CancelledError()
                    if self._quarantine or self._closed:
                        raise BufferDrainError("Compression runtime is unavailable")
                    try:
                        self._charge(size)
                    except CompressionCapacityError:
                        if not wait:
                            raise
                    else:
                        self._workspace_bytes += size
                        return WorkspaceReservation(self, size)
                # Only an external execution thread may wait. No pool/runtime
                # lock, scheduler thread, or GPU allocation is held here.
                time.sleep(0.001)
        finally:
            with self._lock:
                self._workspace_waiters -= 1
                self.stats["workspace_wait_seconds"] += time.monotonic() - started

    def is_quarantined(self):
        with self._lock:
            return bool(self._quarantine)

    def decode_workspace_bytes(self, pages, consumer_bytes=0):
        raw = len(pages) * self.layout.page_bytes
        # Output + possible slot-major copy + caller's verification storage.
        # Host inputs also need H2D copies; receive-ring GPU views are external.
        inputs = sum(
            p.nbytes
            for p in pages
            if isinstance(p, HostEncodedPage) or not p.data.is_cuda
        )
        return 2 * raw + inputs + consumer_bytes + 64 * len(pages)

    def _drop(self, key, entry):
        with self._lock:
            if entry.users == 0 and entry.future.done() and not self._quarantine:
                if self._entries.get(key) is entry:
                    del self._entries[key]
                    self._live_bytes -= entry.resident_bytes

    def _release(self, key, entry):
        with self._lock:
            entry.users -= 1
            assert entry.users >= 0
            self._drop(key, entry)

    def _finish_rejected_encode(self, future, batch):
        """Propagate worker rejection to every consumer of a queued encode."""
        if future.cancelled():
            error = concurrent.futures.CancelledError()
        else:
            error = future.exception()
        if error is None:
            return
        with self._lock:
            for key, _, entry in batch:
                if not entry.future.done():
                    entry.future.set_exception(error)
                self._drop(key, entry)

    def acquire_pages(self, refs, indices, ready_event=None, priority=1):
        if len(refs) != len(indices):
            raise ValueError("Page reference/index count mismatch")
        leases, missing = [], []
        sources = []
        try:
            with self._lock:
                if self._closed or self._quarantine:
                    raise RuntimeError("Compression runtime is unavailable")
                if (
                    self._jobs.qsize()
                    + (len(refs) + self.batch_pages - 1) // self.batch_pages
                    > 256
                ):
                    raise CompressionCapacityError("Compression task queue is full")
                for ref, index in zip(refs, indices):
                    key = (int(ref), self.mode)
                    entry = self._entries.get(key)
                    if entry is None and self.provider is not None:
                        from .provider import RepresentationSpec

                        lease = self.provider.acquire(
                            int(ref),
                            RepresentationSpec(
                                self.layout.tag, self.mode, self.force, self.verify
                            ),
                        )
                        if lease is not None:
                            if self.force and lease.future.result().encoding != "lz4":
                                lease.close()
                                raise ValueError(
                                    "Forced LZ4 cannot reuse a raw L2 object"
                                )
                            self.stats["reused_host_pages"] += 1
                            if self.trace_reuse:
                                sources.append([int(ref), "host"])
                            leases.append(lease)
                            continue
                    if entry is None:
                        source = "new"
                        entry = _Entry(concurrent.futures.Future())
                        self._entries[key] = entry
                        missing.append((key, int(index), entry))
                        if self.trace_reuse:
                            sources.append([int(ref), "new"])
                    else:
                        source = "inflight"
                        if self.trace_reuse:
                            sources.append([int(ref), "inflight"])
                        self.stats["reused_inflight_pages"] += 1
                        if entry.job is not None and entry.job[0] > priority:
                            with self._jobs.mutex:
                                entry.job[0] = priority
                                heapq.heapify(self._jobs.queue)
                    entry.users += 1
                    leases.append(
                        Lease(
                            entry.future,
                            lambda k=key, e=entry: self._release(k, e),
                            source=source,
                        )
                    )
                for start in range(0, len(missing), self.batch_pages):
                    batch = missing[start : start + self.batch_pages]
                    # All admission checks ran before creating any entries.
                    job = [
                        priority,
                        next(self._sequence),
                        concurrent.futures.Future(),
                        lambda b=batch: self._encode(b, ready_event),
                        time.monotonic(),
                    ]
                    for _, _, entry in batch:
                        entry.job = job
                    job[2].add_done_callback(
                        lambda future, b=batch: self._finish_rejected_encode(future, b)
                    )
                    self._jobs.put(job)
                    for _, _, entry in batch:
                        entry.submitted = True
            if self.trace_reuse:
                logger.info(
                    "KV_COMPRESSION_ACQUIRE %s",
                    json.dumps(
                        {
                            "priority": priority,
                            "sources": sources,
                        }
                    ),
                )
            return leases
        except Exception as exc:
            # Unqueued entries must become terminal; already queued jobs retain
            # their native source references until the consumer drains them.
            for key, _, entry in missing:
                if not entry.submitted and not entry.future.done():
                    entry.future.set_exception(exc)
            failure = None
            for lease in leases:
                try:
                    lease.future.result()
                except BufferDrainError as drain_exc:
                    failure = drain_exc
                except Exception:
                    pass
            if failure is not None:
                with self._lock:
                    self._quarantine.append(leases)
                raise failure from exc
            for lease in leases:
                lease.close()
            raise

    def _encode(self, batch, ready_event):
        with self._lock:
            for _, _, entry in batch:
                entry.job = None
        raw = outputs = None
        charge = len(batch) * (2 * self.layout.page_bytes + self.output_bound)
        charged = False
        started = time.perf_counter()
        try:
            self._charge(charge)
            charged = True
            events = (
                [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                if self.stream is not None
                else []
            )
            if events:
                events[0].record(self.stream)
            if ready_event is not None:
                self.stream.wait_event(ready_event)
            if events:
                events[1].record(self.stream)
            pack_start = time.perf_counter()
            raw = self.layout.pack_pages([index for _, index, _ in batch])
            self.stats["pack_host_seconds"] += time.perf_counter() - pack_start
            if events:
                events[2].record(self.stream)
            encode_start = time.perf_counter()
            if self.backend is None:
                lengths = [self.layout.page_bytes] * len(batch)
                outputs = []
            else:
                outputs = [
                    torch.empty(
                        self.output_bound, dtype=torch.uint8, device=self.device
                    )
                    for _ in batch
                ]
                lengths = self.backend.compress_batch(list(raw.unbind(0)), outputs)
                if len(lengths) != len(batch):
                    raise ValueError("Compression backend returned an incomplete batch")
            self.stats["encode_host_seconds"] += time.perf_counter() - encode_start
            if events:
                events[3].record(self.stream)
            self.drain()
            if events:
                self.stats["ready_wait_gpu_ms"] += events[0].elapsed_time(events[1])
                self.stats["pack_gpu_ms"] += events[1].elapsed_time(events[2])
                self.stats["encode_gpu_ms"] += events[2].elapsed_time(events[3])
            if self.backend is not None:
                self.stats["attempted_raw_bytes"] += raw.numel()
                self.stats["attempted_encoded_bytes"] += sum(lengths)
                self.stats["attempted_capacity_bytes"] += len(batch) * self.output_bound
                self.stats["nonshrinking_pages"] += sum(
                    n >= self.layout.page_bytes for n in lengths
                )
                self.stats["insufficient_saving_pages"] += sum(
                    n < self.layout.page_bytes <= n + 256 for n in lengths
                )
                if os.environ.get(
                    "SGLANG_KV_COMPRESSION_TRACE_LENGTHS", "0"
                ).lower() in ("1", "true", "yes", "y"):
                    logger.info(
                        "KV_COMPRESSION_LENGTHS %s",
                        json.dumps(
                            {
                                "first_ref": batch[0][0][0],
                                "pages": len(batch),
                                "raw_page_bytes": self.layout.page_bytes,
                                "capacity_per_page": self.output_bound,
                                "actual_lengths": lengths,
                                "force": self.force,
                            }
                        ),
                    )
            pages = []
            for i, length in enumerate(lengths):
                # Include an allowance for the per-page descriptor/alignment.
                use_lz4 = self.backend is not None and (
                    self.force or length + 256 < self.layout.page_bytes
                )
                data = outputs[i][:length] if use_lz4 else raw[i].clone()
                pages.append(
                    EncodedPage(
                        data, "lz4" if use_lz4 else "raw", self.layout.page_bytes
                    )
                )
            self.drain()
            with self._lock:
                self._live_bytes -= charge
                charged = False
                for (key, _, entry), page in zip(batch, pages):
                    entry.resident_bytes = page.data.untyped_storage().nbytes()
                    self._live_bytes += entry.resident_bytes
                    self.stats["encoded_pages"] += 1
                    self.stats["raw_fallback_pages"] += int(
                        self.backend is not None and page.encoding == "raw"
                    )
                    if not entry.future.done():
                        entry.future.set_result(page)
                    self._drop(key, entry)
                self.stats["encode_seconds"] += time.perf_counter() - started
        except BaseException as exc:
            failure = exc
            try:
                self.drain()
            except BufferDrainError as drain_exc:
                failure = drain_exc
            with self._lock:
                if isinstance(failure, BufferDrainError):
                    self._quarantine.append((raw, outputs, batch))
                elif charged:
                    self._live_bytes -= charge
                for key, _, entry in batch:
                    if not entry.future.done():
                        entry.future.set_exception(failure)
                    self._drop(key, entry)

    def decode_pages(self, pages, *, consumer_bytes=0, reservation=None):
        """Worker-only. Return an owned result; caller closes after GPU use."""
        required = self.decode_workspace_bytes(pages, consumer_bytes)
        if reservation is None:
            reservation = self.reserve_workspace(required)
        elif (
            reservation.runtime is not self
            or reservation.closed
            or reservation.size < required
        ):
            raise ValueError("Invalid decode workspace reservation")
        try:
            raw = self._decode_pages(pages)
            return DecodedPages(self, raw, reservation)
        except BaseException as exc:
            if isinstance(exc, BufferDrainError):
                reservation.quarantine(exc, pages)
            else:
                reservation.release()
            raise

    def _decode_pages(self, pages):
        # L2 restore owns an enclosing reservation through unpack + verification.
        with materialize_pages(pages) as contiguous:
            return self._decode_contiguous_pages(contiguous)

    def _decode_contiguous_pages(self, pages):
        """Worker-only; return page-major bytes after all accesses complete."""
        raw = torch.empty(
            (len(pages), self.layout.page_bytes), dtype=torch.uint8, device=self.device
        )
        sources, outputs = [], []
        keepalive = []
        events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            if self.stream is not None
            else []
        )
        copy_start = time.perf_counter()
        try:
            if events:
                events[0].record(self.stream)
            for i, page in enumerate(pages):
                if page.raw_bytes != self.layout.page_bytes:
                    raise ValueError("Cached page layout mismatch")
                src = page.data.to(self.device, non_blocking=True)
                keepalive.append(src)
                if page.encoding == "raw":
                    raw[i].copy_(src, non_blocking=True)
                elif page.encoding == "lz4" and self.backend is not None:
                    sources.append(src)
                    outputs.append(raw[i])
                else:
                    raise ValueError("Unsupported cached page encoding")
            self.stats["restore_copy_host_seconds"] += time.perf_counter() - copy_start
            if events:
                events[1].record(self.stream)
            decompress_start = time.perf_counter()
            if sources:
                self.backend.decompress_batch(sources, outputs)
            self.stats["decompress_host_seconds"] += (
                time.perf_counter() - decompress_start
            )
            if events:
                events[2].record(self.stream)
            self.drain()
            if events:
                self.stats["restore_copy_gpu_ms"] += events[0].elapsed_time(events[1])
                self.stats["decompress_gpu_ms"] += events[1].elapsed_time(events[2])
            return raw
        except BaseException:
            try:
                self.drain()
            except BufferDrainError:
                self._quarantine.append((raw, keepalive, pages))
                raise
            raise

    def restore(self, leases, indices):
        if len(leases) != len(indices):
            raise ValueError("Restore page/index count mismatch")
        if any(not lease.future.done() for lease in leases):
            # A restore running on this sole worker cannot wait for an encode
            # queued behind itself. L2 leases are already READY at acquisition.
            raise RuntimeError("Restore requires completed object leases")
        for start in range(0, len(indices), self.batch_pages):
            subset = leases[start : start + self.batch_pages]
            charge = len(subset) * (2 * self.layout.page_bytes + self.output_bound)
            self._charge(charge)
            raw = None
            events = []
            wrote = False
            try:
                pages = [lease.future.result() for lease in subset]
                expected = [page.raw_sha256 for page in pages]
                if self.verify and any(d is None for d in expected):
                    self.stats["restore_verification_failures"] += 1
                    raise KVVerificationError("Missing L2 source KV checksum")
                raw = self._decode_pages(pages)
                events = (
                    [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    if self.stream is not None
                    else []
                )
                if events:
                    events[0].record(self.stream)
                self.layout.unpack_pages(raw, indices[start : start + len(subset)])
                if self.verify:
                    # Verify actual target pages, not only decompressor output.
                    actual = page_digests(
                        self.layout.pack_pages(indices[start : start + len(subset)])
                    )
                    if actual != expected:
                        self.stats["restore_verification_failures"] += 1
                        raise KVVerificationError(
                            "L2 restored GPU KV checksum mismatch"
                        )
                    self.stats["verified_restore_pages"] += len(subset)
                if events:
                    events[1].record(self.stream)
                wrote = True
            finally:
                # Even an index_copy failure may leave earlier writes running.
                # Establish completion before dropping tensors or reservations.
                try:
                    self.drain()
                except BufferDrainError:
                    self._quarantine.append((raw, leases, indices))
                    raise
                if not self._quarantine:
                    with self._lock:
                        self._live_bytes -= charge
                    if events and wrote:
                        self.stats["writeback_gpu_ms"] += events[0].elapsed_time(
                            events[1]
                        )

    def quarantine(self, error, owners=()):
        """Stop all consumers after an external copy stream cannot be drained."""
        with self._lock:
            self._quarantine.append((error, owners))

    def idle(self):
        with self._lock:
            return (
                not self._entries
                and not self._running
                and self._jobs.empty()
                and not self._quarantine
                and not self._workspace_bytes
                and not self._workspace_waiters
            )

    def has_pending_work(self):
        with self._lock:
            return bool(
                self._running
                or not self._jobs.empty()
                or self._workspace_bytes
                or self._workspace_waiters
            )

    def snapshot(self):
        with self._lock:
            return dict(
                self.stats,
                resident_bytes=self._live_bytes,
                workspace_bytes=self._workspace_bytes,
                workspace_waiters=self._workspace_waiters,
                inflight_objects=len(self._entries),
                queued_tasks=self._jobs.qsize(),
                running_tasks=self._running,
                quarantined=len(self._quarantine),
                active_job=dict(self._active_job) if self._active_job else None,
            )

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._jobs.put([100, next(self._sequence), None, None, time.monotonic()])
        self._worker.join()

    def transient_refs(self, count):
        return new_page_refs(count)
