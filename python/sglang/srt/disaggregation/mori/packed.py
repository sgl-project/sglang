"""Opt-in sender-side page packing for heterogeneous-TP MoRI KV writes.

Uses the common Triton gather, as the NIXL packing implementation in
sgl-project/sglang#44615 does. Packed pages already have the destination's
local-head layout and are written directly into the destination KV pool.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from sglang.srt.disaggregation.packed_layout import (
    PACKED_LAYOUT_VERSION,
)
from sglang.srt.disaggregation.packed_layout import (
    describe_packable_pool as describe_packable_pool,
)


@dataclass(frozen=True)
class PackedSlice:
    page_size: int
    src_page_bytes: int
    dst_page_bytes: int
    head_offset_bytes: int

    def validate(self) -> None:
        if min(self.page_size, self.src_page_bytes, self.dst_page_bytes) <= 0:
            raise ValueError("Packing requires positive page sizes")
        if self.src_page_bytes % self.page_size or self.dst_page_bytes % self.page_size:
            raise ValueError("KV page byte lengths must contain whole token rows")
        src_row = self.src_page_bytes // self.page_size
        dst_row = self.dst_page_bytes // self.page_size
        if self.head_offset_bytes < 0 or self.head_offset_bytes + dst_row > src_row:
            raise ValueError("Packed head slice is outside the source token row")


def negotiate_packed_slice(
    local, remote, component, *, src_tp, dst_tp, src_rank, dst_rank, total_heads
):
    """Legacy, replicated-head, dtype/layout and unsupported TP cases fall back."""
    if not isinstance(local, dict) or not isinstance(remote, dict):
        return None
    if (
        local.get("version") != PACKED_LAYOUT_VERSION
        or remote.get("version") != PACKED_LAYOUT_VERSION
    ):
        return None
    if not (0 < src_tp < dst_tp <= total_heads) or dst_tp % src_tp:
        return None
    if total_heads % src_tp or total_heads % dst_tp:
        return None
    if not (0 <= src_rank < src_tp and 0 <= dst_rank < dst_tp):
        return None
    if dst_rank // (dst_tp // src_tp) != src_rank:
        return None
    source, destination = local.get(component), remote.get(component)
    if not isinstance(source, dict) or not isinstance(destination, dict):
        return None
    if (
        source.get("heads") != total_heads // src_tp
        or destination.get("heads") != total_heads // dst_tp
    ):
        return None
    keys = ("page_size", "head_dim", "element_bytes", "storage_dtype", "logical_dtype")
    if any(
        source.get(key) is None or source.get(key) != destination.get(key)
        for key in keys
    ):
        return None
    try:
        page_size = int(source["page_size"])
        head_bytes = int(source["head_dim"]) * int(source["element_bytes"])
        if head_bytes <= 0:
            return None
        if source["page_bytes"] != page_size * source["heads"] * head_bytes:
            return None
        if destination["page_bytes"] != page_size * destination["heads"] * head_bytes:
            return None
        head_start = (dst_rank % (dst_tp // src_tp)) * destination["heads"]
        layout = PackedSlice(
            page_size,
            source["page_bytes"],
            destination["page_bytes"],
            head_start * head_bytes,
        )
        layout.validate()
    except (KeyError, TypeError, ValueError):
        return None
    return layout


def packed_page_offsets(num_tensors: int, dst_pages, page_bytes: int):
    """Return offsets relative to one staging MR and each destination MR."""
    pages = [int(page) for page in dst_pages]
    if num_tensors <= 0 or page_bytes <= 0 or any(page < 0 for page in pages):
        raise ValueError("Invalid packed page descriptor metadata")
    count = len(pages)
    return (
        [
            [(tensor * count + page) * page_bytes for page in range(count)]
            for tensor in range(num_tensors)
        ],
        [[page * page_bytes for page in pages] for _ in range(num_tensors)],
        [[page_bytes] * count for _ in range(num_tensors)],
    )


# A failed teardown must not free a buffer while a NIC can still read it.
# Explicitly retain such senders, including their IOEngine and registration.
_QUARANTINED_SENDERS = []


class _UnknownSubmissionStatus:
    """Prevent an abort ACK when an exception hid possibly submitted writes."""

    untracked_submission = True

    def InProgress(self):
        return True

    def Failed(self):
        return True

    def Message(self):
        return "MoRI packed submission raised without returning completion statuses"

    def Wait(self):
        raise RuntimeError(self.Message())


class MoriPackedKVSender:
    """One registered pack buffer owned by one transfer worker thread.

    Every returned status is recorded before another tensor is submitted. A
    timeout never makes the pack buffer reusable: the existing manager drainer
    can settle those same statuses, after which a later call may reuse it.

    Source KV must be ready on the caller's current GPU stream. A producer on
    an independent stream requires an external completion event (the manager's
    wait_event contract). The private gather stream waits on the caller stream;
    this sender is created and used outside graph capture.
    """

    def __init__(
        self,
        *,
        engine,
        gpu_id: int,
        size_bytes: int,
        wait_completion: Callable,
        record_statuses: Callable,
        buffer=None,
        memory_location=None,
        gather: Optional[Callable] = None,
    ):
        if size_bytes <= 0:
            raise ValueError("MoRI KV pack buffer size must be positive")
        if buffer is None:
            from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer

            buffer = StagingBuffer(size_bytes, f"cuda:{gpu_id}", gpu_id)
        if memory_location is None:
            from mori.io import MemoryLocationType

            memory_location = MemoryLocationType.GPU
        self.engine = engine
        self.gpu_id = gpu_id
        self.buffer = buffer
        self.memory_desc = engine.register_memory(
            buffer.get_ptr(), buffer.get_size(), gpu_id, memory_location
        )
        self.wait_completion = wait_completion
        self.record_statuses = record_statuses
        self._gather_override = gather
        self._src_pointers = {}
        self._inflight = []
        self._unknown_submission = False
        self._failed_submission = False
        self._closed = False
        self._operation_lock = threading.Lock()
        self._gpu_gather_unsettled = False
        self._gather_ownership = []

    def _wait(self):
        if self._closed:
            raise RuntimeError("MoRI packed sender is closed")
        if self._unknown_submission:
            raise RuntimeError(
                "MoRI pack buffer quarantined after ambiguous submission"
            )
        if self._failed_submission:
            raise RuntimeError("MoRI pack buffer quarantined after failed submission")
        if self._gpu_gather_unsettled:
            raise RuntimeError("MoRI pack buffer has an unsettled GPU gather")
        if not self._inflight:
            return
        statuses = self._inflight
        # Propagate completion exceptions without dropping the held statuses.
        reason = self.wait_completion(statuses)
        if any(status.InProgress() for status in statuses):
            raise RuntimeError(reason or "MoRI packed writes have not settled")
        if any(status.Failed() for status in statuses):
            # A failed status can cover partially posted WRs. !InProgress()
            # does not prove the NIC stopped reading the registered source.
            self._failed_submission = True
            if self not in _QUARANTINED_SENDERS:
                _QUARANTINED_SENDERS.append(self)
            raise RuntimeError(reason or "MoRI packed write failed")
        self._inflight = []
        if reason is not None:
            raise RuntimeError(reason or "MoRI packed write failed")

    def _gather(self, src_ptrs, pages, layout: PackedSlice):
        if self._gpu_gather_unsettled:
            raise RuntimeError("MoRI pack buffer has an unsettled GPU gather")
        if self._gather_override is not None:
            self._gpu_gather_unsettled = True
            self._gather_override(src_ptrs, pages, layout, self.buffer)
            self._gpu_gather_unsettled = False
            return
        import torch
        import triton

        from sglang.srt.disaggregation.common.staging_buffer import (
            _fused_gather_to_staging_kernel,
        )

        torch.cuda.set_device(self.gpu_id)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MoRI KV packing cannot submit gathers during graph capture"
            )
        stream = self.buffer.get_gather_stream()
        # A source produced on a different independent stream still requires
        # the manager's external wait_event. Honor this caller's current stream
        # before uploading metadata or reading KV on the private gather stream.
        stream.wait_stream(torch.cuda.current_stream(self.gpu_id))
        self._gpu_gather_unsettled = True
        self._gather_ownership = []
        token_count = len(pages) * layout.page_size
        dst_row_bytes = layout.dst_page_bytes // layout.page_size
        tensor_bytes = len(pages) * layout.dst_page_bytes
        with torch.cuda.stream(stream):
            page_tensor = torch.tensor(
                pages, dtype=torch.int64, device=self.buffer.buffer.device
            )
            self._gather_ownership.append(page_tensor)
            key = tuple(src_ptrs)
            pointers = self._src_pointers.get(key)
            if pointers is None:
                pointers = torch.tensor(
                    key, dtype=torch.int64, device=self.buffer.buffer.device
                )
                self._src_pointers[key] = pointers
            self._gather_ownership.append(pointers)
            _fused_gather_to_staging_kernel[
                (len(src_ptrs), triton.cdiv(tensor_bytes, 1024))
            ](
                pointers,
                page_tensor,
                self.buffer.buffer,
                token_count,
                layout.src_page_bytes // layout.page_size,
                layout.head_offset_bytes,
                tensor_bytes,
                dst_row_bytes,
                layout.page_size,
                1024,
            )
        # MoRI's CPU submission must not observe unfinished GPU writes.
        stream.synchronize()
        self._gpu_gather_unsettled = False
        self._gather_ownership = []

    def send(self, **kwargs):
        with self._operation_lock:
            return self._send(**kwargs)

    def _send(
        self,
        *,
        src_ptrs,
        dst_descs,
        src_pages,
        dst_pages,
        src_page_bytes: int,
        dst_page_bytes: int,
        page_size: int,
        head_offset_bytes: int,
    ):
        layout = PackedSlice(
            page_size, src_page_bytes, dst_page_bytes, head_offset_bytes
        )
        layout.validate()
        if len(src_ptrs) != len(dst_descs) or not len(src_ptrs):
            raise ValueError("Packing requires paired nonempty tensor lists")
        if len(src_pages) != len(dst_pages):
            raise ValueError("Packing requires paired source and destination pages")
        if any(int(page) < 0 for page in src_pages):
            raise ValueError("Negative source page")
        # Validate before launching a gather, even if the first batch is empty.
        packed_page_offsets(len(src_ptrs), dst_pages, dst_page_bytes)
        batch_pages = self.buffer.get_size() // (len(src_ptrs) * dst_page_bytes)
        if batch_pages == 0:
            raise ValueError("MoRI pack buffer cannot hold one page across all tensors")
        result = []
        for start in range(0, len(src_pages), batch_pages):
            self._wait()
            end = min(start + batch_pages, len(src_pages))
            self._gather(src_ptrs, src_pages[start:end], layout)
            local, remote, sizes = packed_page_offsets(
                len(src_ptrs), dst_pages[start:end], dst_page_bytes
            )
            for tensor, dst_desc in enumerate(dst_descs):
                # An exception without statuses cannot prove the NIC has stopped.
                self._unknown_submission = True
                try:
                    statuses = self.engine.batch_write(
                        [self.memory_desc],
                        [local[tensor]],
                        [dst_desc],
                        [remote[tensor]],
                        [sizes[tensor]],
                        [self.engine.allocate_transfer_uid()],
                    )
                    statuses = list(statuses)
                    if not statuses:
                        raise RuntimeError(
                            "MoRI packed write returned no completion statuses"
                        )
                except Exception:
                    unknown = _UnknownSubmissionStatus()
                    self._inflight.append(unknown)
                    self.record_statuses([unknown])
                    raise
                self._unknown_submission = False
                self._inflight.extend(statuses)
                self.record_statuses(statuses)
                result.extend(statuses)
            self._wait()
        return result

    def close(self) -> bool:
        """Deregister only when all writes are known quiescent."""
        if not self._operation_lock.acquire(blocking=False):
            if self not in _QUARANTINED_SENDERS:
                _QUARANTINED_SENDERS.append(self)
            return False
        try:
            return self._close()
        finally:
            self._operation_lock.release()

    def _close(self) -> bool:
        if self._closed:
            return True
        # Failure may have arrived after a completion callback raised, before
        # _wait observed it. Quarantine here too rather than deregistering.
        if any(status.Failed() for status in self._inflight):
            self._failed_submission = True
        if (
            self._gpu_gather_unsettled
            or self._unknown_submission
            or self._failed_submission
            or any(s.InProgress() for s in self._inflight)
        ):
            if self not in _QUARANTINED_SENDERS:
                _QUARANTINED_SENDERS.append(self)
            return False
        self.engine.deregister_memory(self.memory_desc)
        self._closed = True
        self._src_pointers.clear()
        if self in _QUARANTINED_SENDERS:
            _QUARANTINED_SENDERS.remove(self)
        return True
