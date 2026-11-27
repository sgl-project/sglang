"""Packs one decode rank's draft KV heads into registered GPU staging, page by page.

Draft KV is [slot, head, head_dim]: with fewer heads per decode rank, the bytes a
decode rank needs are strided per token. Packing them on the prefill GPU turns one
RDMA write per token into one per page.
"""

from __future__ import annotations

import logging
import queue
from typing import Callable, List, Optional, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.transport.base import BufferKey, Copy

# Arbitrary; bounds the staging memory, not the request length.
_MAX_PAGES_PER_BATCH = 256
_NUM_BUFFERS = 4

logger = logging.getLogger(__name__)


class DraftEntryJob(msgspec.Struct, frozen=True):
    entry: int  # index into the draft tensor list
    dst_key: BufferKey
    head_start: int
    num_heads: int


class _StagingBuffer:
    def __init__(self, *, nbytes: int, device: torch.device):
        self.data = torch.empty(nbytes, dtype=torch.uint8, device=device)
        self.stream = torch.cuda.Stream(device=device)
        # Set once the buffer is registered with the Transport.
        self.key: Optional[BufferKey] = None


class DraftHeadPacker:
    def __init__(
        self,
        *,
        tensors: List[torch.Tensor],
        page_size: int,
        window_pages: int,
        register: Callable[[List[Tuple[int, int]]], List[BufferKey]],
    ):
        self._tensors = tensors
        self._page_size = page_size
        self._device = tensors[0].device
        # A windowed draft sends at most window_pages per request, so a larger
        # staging buffer would sit unused; 0 means every page is sent.
        self._pages_per_batch = (
            min(window_pages, _MAX_PAGES_PER_BATCH)
            if window_pages > 0
            else _MAX_PAGES_PER_BATCH
        )
        # Sized for a decode rank that keeps every local head, the upper bound.
        self._batch_bytes = self._pages_per_batch * sum(
            t[0].nbytes * page_size for t in tensors
        )
        self._free: queue.Queue[_StagingBuffer] = queue.Queue()
        buffers = [
            _StagingBuffer(nbytes=self._batch_bytes, device=self._device)
            for _ in range(_NUM_BUFFERS)
        ]
        # The engine raises when it rejects a region; there is no status to check.
        keys = register([(b.data.data_ptr(), self._batch_bytes) for b in buffers])
        for b, key in zip(buffers, keys, strict=True):
            b.key = key
            self._free.put(b)

    def send(
        self,
        *,
        src_page_indices: npt.NDArray,
        dst_page_indices: npt.NDArray,
        jobs: List[DraftEntryJob],
        transfer: Callable[[List[Copy]], int],
    ) -> int:
        buffer = self._free.get()
        try:
            for start in range(0, len(src_page_indices), self._pages_per_batch):
                end = start + self._pages_per_batch
                copies = self._pack(
                    buffer=buffer,
                    src_pages=src_page_indices[start:end],
                    dst_pages=dst_page_indices[start:end],
                    jobs=jobs,
                )
                ret = transfer(copies)
                if ret != 0:
                    return ret
            return 0
        except torch.OutOfMemoryError as e:
            # Runs on the transfer executor: a raise would stop the transfer
            # worker, so fail only this request.
            logger.error(f"packing draft KV heads ran out of GPU memory: {e}")
            return -1
        finally:
            self._free.put(buffer)

    def _pack(
        self,
        *,
        buffer: _StagingBuffer,
        src_pages: npt.NDArray,
        dst_pages: npt.NDArray,
        jobs: List[DraftEntryJob],
    ) -> List[Copy]:
        num_pages = len(src_pages)
        copies = []
        offset = 0
        with torch.cuda.device(self._device), torch.cuda.stream(buffer.stream):
            pages = torch.from_numpy(src_pages.astype(np.int64)).to(
                self._device, non_blocking=True
            )
            for job in jobs:
                tensor = self._tensors[job.entry]
                whole_pages = tensor.shape[0] // self._page_size * self._page_size
                # [slot, head, head_dim] -> [page, token, head, head bytes]
                paged = (
                    tensor[:whole_pages]
                    .view(torch.uint8)
                    .view(-1, self._page_size, tensor.shape[1], tensor[0, 0].nbytes)
                )
                # Slice heads before gathering so the temporary holds only these heads.
                picked = paged[
                    :, :, job.head_start : job.head_start + job.num_heads
                ].index_select(0, pages)
                page_bytes = picked[0].numel()
                out = buffer.data[offset : offset + num_pages * page_bytes]
                out.view(picked.shape).copy_(picked)
                ranges = np.empty((num_pages, 3), dtype=np.uint64)
                ranges[:, 0] = (
                    offset + np.arange(num_pages, dtype=np.uint64) * page_bytes
                )
                ranges[:, 1] = dst_pages.astype(np.uint64) * page_bytes
                ranges[:, 2] = page_bytes
                copies.append(Copy(src=buffer.key, dst=job.dst_key, ranges=ranges))
                offset += num_pages * page_bytes
            buffer.stream.synchronize()
        return copies
