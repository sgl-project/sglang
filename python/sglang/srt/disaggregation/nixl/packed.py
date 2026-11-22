from __future__ import annotations

import numpy as np
import torch
import triton

from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    _fused_gather_to_staging_kernel,
)
from sglang.srt.disaggregation.nixl.transport import NixlHandle, StridedRun


def _raise_failure(handle: NixlHandle, error_msg: str):
    if handle.error is not None:
        raise handle.error
    raise RuntimeError(error_msg)


def packed_page_indices(*, num_tensors, dst_pages, dst_num_slots):
    """Prepped-dlist indices of one batch: the gather leaves tensor t's j-th page in
    buffer slot t * len(dst_pages) + j; decode slot = tensor * dst_num_slots + page."""
    num_pages = len(dst_pages)
    src = np.arange(num_tensors * num_pages, dtype=np.int64)
    dst = (
        np.arange(num_tensors, dtype=np.int64)[:, None] * dst_num_slots
        + np.asarray(dst_pages, dtype=np.int64)[None, :]
    ).ravel()
    return src, dst


class PackedKVSender:
    def __init__(
        self, *, transport, gpu_id, size_bytes, register_memory, await_handles
    ):
        self.transport = transport
        self.gpu_id = gpu_id
        self.await_handles = await_handles
        # Transfers are built from cached dlists by index, as the unpacked path does:
        # initialize_xfer over raw descriptors costs tens of ms of CPU per batch.
        self._src_handles: dict[int, object] = {}
        self._dst_handles: dict[tuple, object] = {}
        self.buffer = StagingBuffer(
            size_bytes=size_bytes, device=f"cuda:{gpu_id}", gpu_id=gpu_id
        )
        # register_memory(ptr, size) -> the key the buffer is registered under.
        self._buffer_key = register_memory(
            self.buffer.get_ptr(), self.buffer.get_size()
        )
        self.inflight = None
        self._src_pointers: dict[tuple[int, ...], torch.Tensor] = {}

    def _gather(
        self,
        *,
        src_ptrs,
        pages,
        page_size,
        src_token_bytes,
        dst_token_bytes,
        head_offset_bytes,
    ):
        num_tokens = len(pages) * page_size
        tensor_bytes = num_tokens * dst_token_bytes
        torch.cuda.set_device(self.gpu_id)
        stream = self.buffer.get_gather_stream()
        # The transfer worker already waited for the producer; keep uploads off
        # the default stream, which prefill CUDA graph capture may be using.
        with torch.cuda.stream(stream):
            indices = torch.tensor(pages, dtype=torch.int64, device=self.buffer.device)
            pointers = self._src_pointers.get(tuple(src_ptrs))
            if pointers is None:
                pointers = torch.tensor(
                    src_ptrs, dtype=torch.int64, device=self.buffer.device
                )
                self._src_pointers[tuple(src_ptrs)] = pointers
            _fused_gather_to_staging_kernel[
                (len(src_ptrs), triton.cdiv(tensor_bytes, 1024))
            ](
                pointers,
                indices,
                self.buffer.buffer,
                num_tokens,
                src_token_bytes,
                head_offset_bytes,
                tensor_bytes,
                dst_token_bytes,
                page_size,
                1024,
            )
        stream.synchronize()

    def _src_handle(self, page_bytes):
        handle = self._src_handles.get(page_bytes)
        if handle is None:
            slots = self.buffer.get_size() // page_bytes
            run = StridedRun(
                key=self._buffer_key,
                offset=0,
                length=page_bytes,
                stride=page_bytes,
                count=slots,
            )
            handle = self._src_handles[page_bytes] = self.transport.prepare(None, [run])
        return handle

    def _dst_handle(self, remote, dst_keys, page_bytes, dst_num_slots):
        key = (remote.agent_name, tuple(dst_keys), page_bytes, dst_num_slots)
        handle = self._dst_handles.get(key)
        if handle is None:
            runs = [
                StridedRun(
                    key=k,
                    offset=0,
                    length=page_bytes,
                    stride=page_bytes,
                    count=dst_num_slots,
                )
                for k in dst_keys
            ]
            handle = self._dst_handles[key] = self.transport.prepare(remote, runs)
        return handle

    def _wait(self, *, raise_on_failure=True):
        if self.inflight is None:
            return
        settled, failed = self.await_handles([self.inflight], failure_seen=False)
        if not settled:
            # Keep the handle: subsequent requests must not overwrite NIC input.
            raise RuntimeError("NIXL packed transfer has not settled")
        self.inflight = None
        if failed and raise_on_failure:
            raise RuntimeError("NIXL packed transfer failed")

    def send(
        self,
        *,
        remote,
        src_ptrs,
        dst_keys,
        src_pages,
        dst_pages,
        src_page_bytes,
        dst_page_bytes,
        page_size,
        head_offset_bytes,
        dst_num_slots,
        handles,
    ):
        # The gather fills src-shaped bytes; the descriptors address dst-shaped ones.
        if len(src_pages) != len(dst_pages) or len(src_ptrs) != len(dst_keys):
            raise ValueError("NIXL packed transfer requires paired pages and tensors")
        if not len(src_pages):
            return
        pages_per_batch = self.buffer.get_size() // (len(src_ptrs) * dst_page_bytes)
        if pages_per_batch == 0:
            raise ValueError(
                "NIXL pack buffer cannot hold one page across all tensors; "
                "increase SGLANG_NIXL_KV_PACK_BUFFER_SIZE_MB"
            )

        src_handle = self._src_handle(dst_page_bytes)
        dst_handle = self._dst_handle(remote, dst_keys, dst_page_bytes, dst_num_slots)
        # A handle left by an earlier send that failed belongs to that room: wait
        # until it no longer reads the buffer, but do not fail this room for it.
        self._wait(raise_on_failure=False)
        for start in range(0, len(src_pages), pages_per_batch):
            self._wait()
            end = min(start + pages_per_batch, len(src_pages))
            self._gather(
                src_ptrs=src_ptrs,
                pages=src_pages[start:end],
                page_size=page_size,
                src_token_bytes=src_page_bytes // page_size,
                dst_token_bytes=dst_page_bytes // page_size,
                head_offset_bytes=head_offset_bytes,
            )
            src_indices, dst_indices = packed_page_indices(
                num_tensors=len(dst_keys),
                dst_pages=dst_pages[start:end],
                dst_num_slots=dst_num_slots,
            )
            # No notification: the transfer worker announces the chunk once its handles settle.
            (handle,) = self.transport.write_prepared(
                src_handle, src_indices, dst_handle, dst_indices
            )
            if handle.result is not None and handle.result.quiescent:
                _raise_failure(handle, "Failed to create NIXL packed transfer")
            self.inflight = handle
            # Let the caller settle posted writes if a later batch fails.
            handles.append(handle)
            if handle.result is not None:
                _raise_failure(handle, "Failed to post NIXL packed transfer")
        self._wait()
