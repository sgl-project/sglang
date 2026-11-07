from __future__ import annotations

from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool


class QSAIndexerPoolHost(DeepSeekV4PagedHostPool):
    """Page-row mirror of QSA compressed keys, indexed by full-KV token slots."""

    def __init__(
        self,
        buffer_info: IndexKeyBufferInfo,
        *,
        num_host_pages: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        buffer_info.validate()
        # A full-KV page and its compressed groups share ownership. Packing the
        # groups into a byte row lets the existing whole-page kernels move both
        # coordinate spaces using the same full-KV page number.
        super().__init__(
            pool_name="qsa_indexer",
            device_buffers=list(buffer_info.page_views()),
            item_bytes=buffer_info.page_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=buffer_info.page_size,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            allocator_type=allocator_type,
            page_aligned_only=True,
        )
        self.size_per_token = self.get_size_per_token()

    def get_size_per_token(self):
        return self.layer_num * self.item_bytes // self.slot_page_size

    def get_ksize_per_token(self):
        return self.get_size_per_token()
