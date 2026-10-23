from __future__ import annotations

import numpy as np
import torch
import triton

from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    _fused_gather_to_staging_kernel,
)


def packed_page_descriptors(
    *, base_ptr, dst_ptrs, dst_pages, page_bytes, src_gpu_id, dst_gpu_id
):
    offsets = np.arange(len(dst_ptrs) * len(dst_pages), dtype=np.uint64)
    src_addrs = base_ptr + offsets * np.uint64(page_bytes)
    dst_addrs = (
        np.asarray(dst_ptrs, dtype=np.uint64)[:, None]
        + np.asarray(dst_pages, dtype=np.uint64)[None, :] * np.uint64(page_bytes)
    ).ravel()
    sizes = np.full(len(src_addrs), page_bytes, dtype=np.uint64)
    src = np.column_stack((src_addrs, sizes, np.full_like(sizes, src_gpu_id)))
    dst = np.column_stack((dst_addrs, sizes, np.full_like(sizes, dst_gpu_id)))
    return src, dst


class PackedKVSender:
    def __init__(self, *, agent, gpu_id, size_bytes, register_memory, await_handles):
        self.agent = agent
        self.gpu_id = gpu_id
        self.await_handles = await_handles
        self.buffer = StagingBuffer(
            size_bytes=size_bytes, device=f"cuda:{gpu_id}", gpu_id=gpu_id
        )
        register_memory(self.buffer.get_ptr(), self.buffer.get_size())
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

    def _wait(self):
        if self.inflight is None:
            return
        settled, failed = self.await_handles([self.inflight], failure_seen=False)
        if not settled:
            # Keep the handle: subsequent requests must not overwrite NIC input.
            raise RuntimeError("NIXL packed transfer has not settled")
        self.inflight = None
        if failed:
            raise RuntimeError("NIXL packed transfer failed")

    def send(
        self,
        *,
        peer_name,
        src_ptrs,
        dst_ptrs,
        src_pages,
        dst_pages,
        src_page_bytes,
        dst_page_bytes,
        page_size,
        head_offset_bytes,
        dst_gpu_id,
        notif,
        handles,
    ):
        # The gather fills src-shaped bytes; the descriptors address dst-shaped ones.
        if len(src_pages) != len(dst_pages) or len(src_ptrs) != len(dst_ptrs):
            raise ValueError("NIXL packed transfer requires paired pages and tensors")
        if not len(src_pages):
            return
        pages_per_batch = self.buffer.get_size() // (len(src_ptrs) * dst_page_bytes)
        if pages_per_batch == 0:
            raise ValueError(
                "NIXL pack buffer cannot hold one page across all tensors; "
                "increase SGLANG_NIXL_KV_PACK_BUFFER_SIZE_MB"
            )

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
            src, dst = packed_page_descriptors(
                base_ptr=self.buffer.get_ptr(),
                dst_ptrs=dst_ptrs,
                dst_pages=dst_pages[start:end],
                page_bytes=dst_page_bytes,
                src_gpu_id=self.gpu_id,
                dst_gpu_id=dst_gpu_id,
            )
            handle = self.agent.initialize_xfer(
                "WRITE",
                self.agent.get_xfer_descs(src, "VRAM"),
                self.agent.get_xfer_descs(dst, "VRAM"),
                peer_name,
                notif.encode("ascii") if end == len(src_pages) else b"",
            )
            if not handle:
                raise RuntimeError("Failed to create NIXL packed transfer")
            self.inflight = handle
            # Let the caller settle posted writes if a later batch fails.
            handles.append(handle)
            if self.agent.transfer(handle) == "ERR":
                raise RuntimeError("Failed to post NIXL packed transfer")
        self._wait()
