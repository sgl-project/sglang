"""Zero freed KV pages before the allocator hands them out again."""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def kv_page_buffers(pool) -> list[torch.Tensor]:
    """Every page-indexed buffer of ``pool`` as a uint8 [rows, page_bytes] view,
    where row ``p`` holds page ``p``."""
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool

    if type(pool) not in (MLATokenToKVPool, DSATokenToKVPool):
        raise NotImplementedError(
            f"Zeroing KV pages on free does not support {type(pool).__name__}"
        )
    buffers = [
        buf.view(torch.uint8).view(-1, pool.page_size * buf[0].nbytes)
        for buf in pool.kv_buffer
    ]
    if type(pool) is DSATokenToKVPool:
        buffers += [
            buf.view(torch.uint8).view(buf.shape[0], -1)
            for buf in pool.index_key_cache.buffer
            if buf.shape[0] > 0
        ]
    return buffers


class KVPageZeroer:
    def __init__(
        self,
        buffers: Sequence[torch.Tensor],
        wait_streams: Sequence[torch.cuda.Stream] = (),
    ):
        assert buffers, "no KV buffers to zero"
        self.buffers = list(buffers)
        self.num_rows = min(b.shape[0] for b in self.buffers)
        self.device = self.buffers[0].device
        self.is_cuda = self.device.type == "cuda"
        self.wait_streams = list(wait_streams)
        if not self.is_cuda:
            return

        page_bytes = [b.shape[1] for b in self.buffers]
        assert all(n % 8 == 0 for n in page_bytes), page_bytes
        self.buf_ptrs = torch.tensor(
            [b.data_ptr() for b in self.buffers], dtype=torch.int64, device=self.device
        )
        self.buf_page_words = torch.tensor(
            [n // 8 for n in page_bytes], dtype=torch.int64, device=self.device
        )
        self.max_page_words = max(page_bytes) // 8
        self.stream = torch.cuda.Stream(device=self.device)

    @classmethod
    def from_pools(cls, pools, wait_streams=()) -> KVPageZeroer:
        unique = {id(p): p for p in pools if p is not None}
        buffers = [b for p in unique.values() for b in kv_page_buffers(p)]
        return cls(buffers, wait_streams)

    def zero(self, page_ids: torch.Tensor) -> Optional[torch.cuda.Event]:
        """Queue zeroing of ``page_ids``. The pages may be reused once the
        returned event completes (None means already done)."""
        if not self.is_cuda:
            for buf in self.buffers:
                buf[page_ids] = 0
            return None

        from sglang.kernels.ops.kvcache.zero_pages import zero_pages_multi

        # Wait for in-flight forwards: an overlapped forward may still write the
        # last token of a request whose pages are being freed.
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        for stream in self.wait_streams:
            self.stream.wait_stream(stream)
        with torch.cuda.stream(self.stream):
            zero_pages_multi(
                self.buf_ptrs, self.buf_page_words, self.max_page_words, page_ids
            )
        page_ids.record_stream(self.stream)
        event = torch.cuda.Event()
        event.record(self.stream)
        return event
