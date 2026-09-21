"""Experimental transforms around the existing Mooncake staging path."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import threading
import time

import torch
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
from sglang.srt.disaggregation.compression.protocol import (
    BufferDrainError,
    ChunkDescriptor,
)
from sglang.srt.environ import envs
from sglang.srt.kv_compression.layout import KVLayoutAdapter
from sglang.srt.kv_compression.runtime import KVCompressionRuntime
from sglang.srt.kv_compression.store import materialize_pages
from sglang.srt.kv_compression.types import EncodedPage, align_bytes
from sglang.srt.runtime_context import get_schedule

logger = logging.getLogger(__name__)


def digest(tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


class CompressionRuntime:
    """Transport adapter. Shared execution has no Mooncake dependency."""

    def __init__(self, manager, k_buffers, v_buffers, page_size):
        self.manager = manager
        self.mode = manager.compression_mode
        self.force = envs.SGLANG_PD_KV_COMPRESSION_FORCE.get()
        self.layout = KVLayoutAdapter(k_buffers, v_buffers, page_size)
        self.device = self.layout.device
        self.chunk_tokens = get_schedule().chunked_prefill_size
        self.bytes_per_token = self.layout.page_bytes
        self.max_raw_bytes = self.chunk_tokens * self.bytes_per_token
        shared = getattr(manager, "shared_compression_runtime", None)
        if (
            self.force
            and shared is not None
            and (shared.mode != "lz4" or not shared.force)
        ):
            raise ValueError("Forced P/D and L2 must share a forced LZ4 runtime")
        if shared is None or shared.mode != self.mode:
            self.shared = KVCompressionRuntime(
                self.layout,
                self.mode,
                envs.SGLANG_KV_COMPRESSION_WORKSPACE_MB.get() * 1024**2,
                force=self.force,
                verify=envs.SGLANG_PD_KV_COMPRESSION_VERIFY.get(),
            )
            self.shared.provider = getattr(manager, "encoded_kv_provider", None)
        else:
            if shared.force != self.force:
                raise ValueError("P/D and L2 compression force policies differ")
            if shared.layout.tag != self.layout.tag:
                raise ValueError("P/D and L2 compression layout mismatch")
            self.shared = shared
        self.layout_tag = self.layout.tag + f"/chunk={self.chunk_tokens}"
        self.wire_capacity = self.chunk_tokens * align_bytes(
            max(self.bytes_per_token, self.shared.output_bound)
        )
        if 2 * self.max_raw_bytes + self.wire_capacity > self.shared.budget_bytes:
            raise ValueError(
                "Chunk exceeds compression workspace budget; reduce chunk size or raise SGLANG_KV_COMPRESSION_WORKSPACE_MB"
            )
        self.local = threading.local()
        self.outputs = []  # Registered allocation ownership lasts until exit.
        self.transport_failed = False
        self.verify = envs.SGLANG_PD_KV_COMPRESSION_VERIFY.get()
        allocator = getattr(manager._staging_ctx, "allocator", None)
        if allocator is not None and self.wire_capacity > allocator.total_size:
            raise ValueError("Compression chunk cannot fit in Decode staging")
        logger.info(
            "PD_KV_COMPRESSION_CONFIG %s",
            json.dumps(
                {
                    "mode": self.mode,
                    "force": self.force,
                    "verify": self.verify,
                    "shared_l2": self.shared.provider is not None,
                    "chunk_tokens": self.chunk_tokens,
                    "workspace_bytes": self.shared.budget_bytes,
                }
            ),
        )

    def source_digest(self, indices, stream, ready_event):
        """Verify the protected source; failure must also finish source reads."""
        raw = None
        try:
            with torch.cuda.stream(stream):
                if ready_event is not None:
                    stream.wait_event(ready_event)
                raw = self.layout.to_staging_order(self.layout.pack_pages(indices))
                return digest(raw)
        finally:
            try:
                stream.synchronize()
            except Exception as exc:
                self.transport_failed = True
                self.shared.quarantine(exc, (raw, indices, ready_event))
                raise BufferDrainError(
                    "Source KV verification did not drain; restart worker"
                ) from exc

    def encode_pages(self, indices, refs, ready_event, stream):
        if self.transport_failed:
            raise RuntimeError("Compression transport is unavailable; restart worker")
        started = time.perf_counter()
        if refs is None:
            refs = self.shared.transient_refs(len(indices))
        leases = self.shared.acquire_pages(refs, indices, ready_event, priority=1)
        self.local.ref_sources = [lease.source for lease in leases]
        wire = getattr(self.local, "output", None)
        pages = []
        try:
            pages = [lease.future.result() for lease in leases]
            object_wait_ms = (time.perf_counter() - started) * 1000
            wire_prepare_started = time.perf_counter()
            if wire is None:
                wire = torch.empty(
                    self.wire_capacity, dtype=torch.uint8, device=self.device
                )
                self.outputs.append(wire)
                try:
                    self.manager._register_staging_memory(wire.data_ptr(), wire.numel())
                except Exception:
                    # Registration may be partial. Retain this allocation,
                    # but do not allocate another wire buffer on every retry.
                    self.transport_failed = True
                    raise
                self.local.output = wire
            wire_prepare_ms = (time.perf_counter() - wire_prepare_started) * 1000
            assembly_started = time.perf_counter()
            descriptors, end, host_bytes = [], 0, 0
            for page in pages:
                offset = align_bytes(end)
                descriptors.append((offset, page.nbytes, page.encoding))
                end = offset + page.nbytes
            if end > wire.numel():
                raise ValueError("Page payload exceeds registered wire capacity")
            with torch.cuda.stream(stream):
                wire[:end].zero_()  # Never transmit stale alignment padding.
                for start in range(0, len(pages), self.shared.batch_pages):
                    end_batch = start + self.shared.batch_pages
                    with materialize_pages(pages[start:end_batch]) as contiguous:
                        try:
                            for (offset, length, _), page in zip(
                                descriptors[start:end_batch], contiguous
                            ):
                                wire[offset : offset + length].copy_(
                                    page.data, non_blocking=True
                                )
                                if not page.data.is_cuda:
                                    host_bytes += length
                        finally:
                            try:
                                stream.synchronize()
                            except Exception as exc:
                                raise BufferDrainError(
                                    "Payload assembly did not drain"
                                ) from exc
            self.local.host_send_bytes = host_bytes
            logger.info(
                "KV_COMPRESSION_ASSEMBLY %s",
                json.dumps(
                    {
                        "pages": len(pages),
                        "object_wait_ms": object_wait_ms,
                        "wire_prepare_ms": wire_prepare_ms,
                        "assembly_ms": (time.perf_counter() - assembly_started) * 1000,
                        "host_send_bytes": host_bytes,
                        "runtime": self.shared.snapshot(),
                    }
                ),
            )
            return (
                wire[:end],
                tuple(descriptors),
                (time.perf_counter() - started) * 1000,
            )
        finally:
            failure = None
            for lease in leases:
                try:
                    lease.future.result()
                except BufferDrainError as exc:
                    failure = exc
                except Exception:
                    pass
            try:
                stream.synchronize()
            except Exception as exc:
                failure = BufferDrainError(f"Payload assembly did not drain: {exc}")
            if failure is not None:
                self.shared.quarantine(failure, (leases, wire, pages))
                raise failure
            for lease in leases:
                lease.close()


class CompressedDecodeStagingHandler(DecodeStagingHandler):
    """Restore on one background worker, publish completion on scheduler poll."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = self.kv_manager.compression_runtime
        self._restore_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pd-kv-restore"
        )
        self._restore_lock = threading.Lock()
        self._restore_tasks = {}
        self._quarantined = {}
        self._verify_buffer = None
        self.staging_allocator._scatter_stream = torch.cuda.Stream(
            device=self.runtime.device
        )

    def handle_compressed_chunk(
        self, room, chunk_idx, page_start, num_pages, session_id, payload
    ):
        with self._restore_lock:
            req = self._room_to_decode_req.get(room)
            receiver = self._room_to_receiver.get(room)
            if req is None or receiver is None:
                return
            if session_id != receiver.session_id:
                return
            try:
                desc = ChunkDescriptor.from_bytes(payload)
                if desc.nonce != receiver.compression_nonce:
                    return  # Late notification from an older request generation.
                if desc.encoding != "pages":
                    raise ValueError("V2 receiver requires a page manifest")
                infos = receiver.chunk_staging_infos
                if not 0 <= chunk_idx < len(infos):
                    raise ValueError("Chunk has no receiver allocation")
                alloc_id, offset, _, end, pages = infos[chunk_idx]
                if alloc_id < 0 or (room, chunk_idx) in self._restore_tasks:
                    return  # Duplicate notification; never restore/free twice.
                if (
                    pages != num_pages
                    or page_start != chunk_idx * self.runtime.chunk_tokens
                ):
                    raise ValueError(
                        "Chunk page range differs from receiver allocation"
                    )
                desc.validate(
                    nonce=receiver.compression_nonce,
                    raw_bytes=num_pages * self.runtime.bytes_per_token,
                    capacity=end - offset,
                    mode=self.runtime.mode,
                    page_bytes=self.runtime.bytes_per_token,
                )
                if self.runtime.verify and not desc.sha256:
                    raise ValueError("Verified transfer requires a source KV digest")
                if self.runtime.force and any(e != "lz4" for _, _, e in desc.pages):
                    raise ValueError("Forced LZ4 transfer cannot contain raw pages")
                if (
                    req._staging_failed
                    or self.kv_manager.check_status(room) == KVPoll.Failed
                ):
                    return
                future = self._restore_executor.submit(
                    self._restore,
                    req,
                    receiver,
                    offset,
                    page_start,
                    num_pages,
                    desc,
                    time.monotonic(),
                )
                self._restore_tasks[room, chunk_idx] = (future, alloc_id)
            except Exception as exc:
                self._fail(req, exc)

    def _fail(self, req, exc):
        req._staging_failed = True
        room = req.req.bootstrap_room
        if isinstance(exc, BufferDrainError):
            self._quarantined.setdefault(room, req)
        self.kv_manager.record_failure(room, f"KV restoration failed: {exc}")
        self.kv_manager.update_status(room, KVPoll.Failed)
        logger.error("KV restoration failed for room=%s: %s", room, exc)

    def _restore(
        self, req, receiver, offset, page_start, num_pages, desc, submitted_at=None
    ):
        torch.cuda.set_device(self.runtime.device)
        stream = self.staging_allocator._scatter_stream
        queue_ms = (
            (time.monotonic() - submitted_at) * 1000 if submitted_at is not None else 0
        )
        started = time.perf_counter()
        wire = self.staging_allocator.buffer.buffer[offset : offset + desc.wire_bytes]
        begin, layout_begin, decoded, scattered_event = (
            torch.cuda.Event(enable_timing=True) for _ in range(4)
        )
        pages = page_major = raw = future = None
        try:
            with torch.cuda.stream(stream):
                begin.record(stream)
                pages = [
                    EncodedPage(wire[o : o + n], e, self.runtime.bytes_per_token)
                    for o, n, e in desc.pages
                ]
                future = self.runtime.shared.submit(
                    lambda: self.runtime.shared.decode_pages(pages), priority=0
                )
                page_major = future.result()
                restore_wait_ms = (time.perf_counter() - started) * 1000
                layout_begin.record(stream)
                layout_start = time.perf_counter()
                raw = self.runtime.layout.to_staging_order(page_major)
                decoded.record(stream)
                restored = time.perf_counter()
                layout_host_ms = (restored - layout_start) * 1000
                self._scatter_region(
                    offset, page_start, num_pages, req, receiver, staging_view=raw
                )
                scattered_event.record(stream)
                stream.synchronize()
                scattered = time.perf_counter()
                verify_start = time.perf_counter()
                if desc.sha256:
                    self._verify_written_pages(req, page_start, num_pages, desc.sha256)
                event = torch.cuda.Event()
                event.record(stream)
            logger.info(
                "PD_KV_COMPRESSION_RECV %s",
                json.dumps(
                    {
                        "room": req.req.bootstrap_room,
                        "page_start": page_start,
                        "encoding": desc.encoding,
                        "raw_bytes": desc.raw_bytes,
                        "wire_bytes": desc.wire_bytes,
                        "restore_ms": (restored - started) * 1000,
                        "scatter_ms": (scattered - restored) * 1000,
                        "restore_gpu_ms": begin.elapsed_time(decoded),
                        "scatter_gpu_ms": decoded.elapsed_time(scattered_event),
                        "verified": bool(desc.sha256),
                        "lz4_objects": sum(e == "lz4" for _, _, e in desc.pages),
                        "restore_wait_ms": restore_wait_ms,
                        "layout_host_ms": layout_host_ms,
                        "layout_gpu_ms": layout_begin.elapsed_time(decoded),
                        "receiver_queue_ms": queue_ms,
                        "verify_ms": (time.perf_counter() - verify_start) * 1000,
                        "runtime": self.runtime.shared.snapshot(),
                    }
                ),
            )
            return event
        finally:
            # A failed job can have submitted CUDA work. Its future is not
            # quiescent until that work has stopped touching these buffers.
            try:
                stream.synchronize()
            except Exception as exc:
                self._quarantined[req.req.bootstrap_room] = (
                    req,
                    wire,
                    pages,
                    page_major,
                    raw,
                    future,
                )
                raise BufferDrainError("Restore stream did not drain") from exc

    def _verify_written_pages(self, req, page_start, num_pages, expected):
        from sglang.srt.disaggregation.common.staging_buffer import (
            StagingBuffer,
            gather_all_layers_to_staging,
        )

        if self._verify_buffer is None:
            self._verify_buffer = StagingBuffer(
                self.runtime.max_raw_bytes,
                str(self.runtime.device),
                self.runtime.device.index,
            )
        indices = (
            self.scheduler.req_to_token_pool.req_to_token[
                req.req.kv.req_pool_idx, page_start : page_start + num_pages
            ]
            .cpu()
            .numpy()
        )
        nbytes = gather_all_layers_to_staging(
            self.kv_buffer_info["k_buffers"],
            self.kv_buffer_info["v_buffers"],
            indices,
            self._verify_buffer,
            0,
            self.total_kv_heads,
            1,
            self.runtime.device.index,
        )
        if digest(self._verify_buffer.buffer[:nbytes]) != expected:
            raise ValueError("Restored KV pages do not match Prefill KV bytes")

    def advance_scatter(self, decode_req):
        room = decode_req.req.bootstrap_room
        with self._restore_lock:
            for key, (future, alloc_id) in list(self._restore_tasks.items()):
                if key[0] != room or not future.done():
                    continue
                del self._restore_tasks[key]
                try:
                    event = future.result()
                    decode_req._chunk_events.append((event, alloc_id))
                    self._room_to_receiver[room].chunk_staging_infos[key[1]] = (
                        -1,
                        -1,
                        0,
                        -1,
                        0,
                    )
                except Exception as exc:
                    self._fail(decode_req, exc)
        super().advance_scatter(decode_req)

    def unregister_decode_req(self, room):
        with self._restore_lock:
            req = self._room_to_decode_req.pop(room, None)
            receiver = self._room_to_receiver.pop(room, None)
            self._writer_counts.pop(room, None)
            tasks = [
                self._restore_tasks.pop(k)[0]
                for k in list(self._restore_tasks)
                if k[0] == room
            ]
        for future in tasks:
            if not future.cancel():
                try:
                    future.result()  # Teardown only: drain before freeing KV pages.
                except BufferDrainError:
                    self._quarantined.setdefault(room, req)
                except Exception:
                    pass
        if room in self._quarantined:
            raise BufferDrainError("Restore buffers quarantined; restart worker")
        if req is not None:
            self.release_room(room, req, receiver)
        self.kv_manager._staging_ctx.room_receivers.pop(room, None)
        self.kv_manager._staging_ctx.room_bootstrap.pop(room, None)

    def is_idle(self):
        with self._restore_lock:
            return not (
                self._restore_tasks or self._quarantined or self._room_to_decode_req
            )
