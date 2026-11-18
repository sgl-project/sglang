"""Packs one decode rank's draft KV heads into registered GPU staging, page by page.

Draft KV is [slot, head, head_dim]: with fewer heads per decode rank, the bytes a
decode rank needs are strided per token. Packing them on the prefill GPU turns one
RDMA write per token into one per page.
"""

from __future__ import annotations

import queue
from typing import Callable, List, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
import torch

# Arbitrary; bounds the staging memory, not the request length.
_PAGES_PER_BATCH = 256
_NUM_BUFFERS = 4


class DraftEntryJob(msgspec.Struct, frozen=True):
    entry: int  # index into the draft tensor list
    dst_base_ptr: int
    head_start: int
    num_heads: int


class _StagingBuffer:
    def __init__(self, *, nbytes: int, device: torch.device):
        self.data = torch.empty(nbytes, dtype=torch.uint8, device=device)
        self.stream = torch.cuda.Stream(device=device)


class DraftHeadPacker:
    def __init__(
        self,
        *,
        tensors: List[torch.Tensor],
        page_size: int,
        register: Callable[[List[int], List[int]], None],
    ):
        self._tensors = tensors
        self._page_size = page_size
        self._device = tensors[0].device
        # Sized for a decode rank that keeps every local head, the upper bound.
        self._batch_bytes = _PAGES_PER_BATCH * sum(
            t[0].nbytes * page_size for t in tensors
        )
        self._free: queue.Queue[_StagingBuffer] = queue.Queue()
        buffers = [
            _StagingBuffer(nbytes=self._batch_bytes, device=self._device)
            for _ in range(_NUM_BUFFERS)
        ]
        # The engine raises when it rejects a region; there is no status to check.
        register(
            [b.data.data_ptr() for b in buffers], [self._batch_bytes] * len(buffers)
        )
        for b in buffers:
            self._free.put(b)

    def send(
        self,
        *,
        src_page_indices: npt.NDArray,
        dst_page_indices: npt.NDArray,
        jobs: List[DraftEntryJob],
        transfer: Callable[[List[int], List[int], List[int]], int],
    ) -> int:
        buffer = self._free.get()
        try:
            for start in range(0, len(src_page_indices), _PAGES_PER_BATCH):
                end = start + _PAGES_PER_BATCH
                src, dst, lengths = self._pack(
                    buffer=buffer,
                    src_pages=src_page_indices[start:end],
                    dst_pages=dst_page_indices[start:end],
                    jobs=jobs,
                )
                ret = transfer(src, dst, lengths)
                if ret != 0:
                    return ret
            return 0
        finally:
            self._free.put(buffer)

    def _pack(
        self,
        *,
        buffer: _StagingBuffer,
        src_pages: npt.NDArray,
        dst_pages: npt.NDArray,
        jobs: List[DraftEntryJob],
    ) -> Tuple[List[int], List[int], List[int]]:
        num_pages = len(src_pages)
        src, dst, lengths = [], [], []
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
                picked = paged.index_select(0, pages)[
                    :, :, job.head_start : job.head_start + job.num_heads
                ]
                page_bytes = picked[0].numel()
                out = buffer.data[offset : offset + num_pages * page_bytes]
                out.view(picked.shape).copy_(picked)
                base = buffer.data.data_ptr() + offset
                src.extend(base + np.arange(num_pages, dtype=np.int64) * page_bytes)
                dst.extend(job.dst_base_ptr + dst_pages.astype(np.int64) * page_bytes)
                lengths.extend([page_bytes] * num_pages)
                offset += num_pages * page_bytes
            buffer.stream.synchronize()
        return [int(x) for x in src], [int(x) for x in dst], lengths
