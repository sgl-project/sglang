"""Sender-only SWA page packing; uses the existing Mooncake sync executor.

No receiver staging protocol or MoRI transport dependency. A nonzero native
sync return does not prove the writes stopped: quarantine instead of reusing
scratch, and let the manager poison the room's deferred abort acknowledgement.
"""

import threading

_QUARANTINED_SENDERS = []


class MooncakePackedSwaSender:
    def __init__(
        self, engine, executor, gpu_id, size_bytes, *, buffer=None, gather=None
    ):
        if size_bytes <= 0:
            raise ValueError("Packing buffer must be positive")
        if buffer is None:
            from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer

            buffer = StagingBuffer(size_bytes, f"cuda:{gpu_id}", gpu_id)
        self.engine, self.executor, self.gpu_id = engine, executor, gpu_id
        self.buffer = buffer
        self.size_bytes = size_bytes
        self._gather_override = gather
        self._lock = threading.Lock()
        self._pending = []
        self._gpu_unsettled = False
        self._ownership = []
        self._pointers = {}
        self._quarantined = False
        self._closed = False
        engine.register(buffer.get_ptr(), size_bytes)

    def _quarantine(self):
        self._quarantined = True
        if self not in _QUARANTINED_SENDERS:
            _QUARANTINED_SENDERS.append(self)

    def _gather(self, src_ptrs, pages, layout):
        if self._gather_override is not None:
            self._gpu_unsettled = True
            self._gather_override(src_ptrs, pages, layout, self.buffer)
            self._gpu_unsettled = False
            return
        import torch
        import triton

        from sglang.srt.disaggregation.common.staging_buffer import (
            _fused_gather_to_staging_kernel,
        )

        torch.cuda.set_device(self.gpu_id)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mooncake SWA packing cannot gather during graph capture"
            )
        stream = self.buffer.get_gather_stream()
        # Independent producer streams must be ordered by the manager's existing
        # kv_chunk.wait_event before entering this method.
        stream.wait_stream(torch.cuda.current_stream(self.gpu_id))
        self._gpu_unsettled = True
        with torch.cuda.stream(stream):
            page_ids = torch.tensor(
                pages, dtype=torch.int64, device=self.buffer.buffer.device
            )
            self._ownership.append(page_ids)
            key = tuple(src_ptrs)
            pointers = self._pointers.get(key)
            if pointers is None:
                pointers = torch.tensor(
                    key, dtype=torch.int64, device=self.buffer.buffer.device
                )
                self._pointers[key] = pointers
            self._ownership.append(pointers)
            tensor_bytes = len(pages) * layout.dst_page_bytes
            _fused_gather_to_staging_kernel[
                (len(src_ptrs), triton.cdiv(tensor_bytes, 1024))
            ](
                pointers,
                page_ids,
                self.buffer.buffer,
                len(pages) * layout.page_size,
                layout.src_page_bytes // layout.page_size,
                layout.head_offset_bytes,
                tensor_bytes,
                layout.dst_page_bytes // layout.page_size,
                layout.page_size,
                1024,
            )
        stream.synchronize()
        self._gpu_unsettled = False
        self._ownership.clear()

    def send(self, session_id, src_ptrs, dst_ptrs, src_pages, dst_pages, layout):
        with self._lock:
            if (
                self._closed
                or self._quarantined
                or self._gpu_unsettled
                or self._pending
            ):
                raise RuntimeError("Mooncake packed sender is not reusable")
            layout.validate()
            count = len(src_ptrs)
            if not count or count != len(dst_ptrs) or len(src_pages) != len(dst_pages):
                raise ValueError("Unpaired packed transfer")
            capacity = self.size_bytes // (count * layout.dst_page_bytes)
            if capacity < 1:
                raise ValueError("Packing buffer cannot hold one page per tensor")
            try:
                for start in range(0, len(src_pages), capacity):
                    pages = list(src_pages[start : start + capacity])
                    dst = list(dst_pages[start : start + capacity])
                    self._gather(src_ptrs, pages, layout)
                    tensor_bytes = len(pages) * layout.dst_page_bytes
                    for tensor, ptr in enumerate(dst_ptrs):
                        local = [
                            self.buffer.get_ptr()
                            + tensor * tensor_bytes
                            + i * layout.dst_page_bytes
                            for i in range(len(pages))
                        ]
                        remote = [
                            int(ptr) + int(page) * layout.dst_page_bytes for page in dst
                        ]
                        lengths = [layout.dst_page_bytes] * len(pages)
                        future = self.executor.submit(
                            self.engine.batch_transfer_sync,
                            session_id,
                            local,
                            remote,
                            lengths,
                        )
                        self._pending.append(future)
                    failed = False
                    for future in self._pending:
                        try:
                            failed = (future.result() != 0) or failed
                        except Exception:
                            failed = True
                    if failed:
                        raise RuntimeError(
                            "Mooncake packed sync failed; native writes may remain active"
                        )
                    self._pending.clear()
            except BaseException:
                # Retain completed failures too: Python future completion says
                # nothing about native timeout cancellation or GPU sync failure.
                self._quarantine()
                raise
            return 0

    def close(self):
        if not self._lock.acquire(blocking=False):
            self._quarantine()
            return False
        try:
            if self._quarantined or self._gpu_unsettled or self._pending:
                self._quarantine()
                return False
            if not self._closed:
                self.engine.deregister(self.buffer.get_ptr())
                self._closed = True
                self._pointers.clear()
            return True
        finally:
            self._lock.release()
