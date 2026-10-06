from __future__ import annotations

import torch

from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


class QSAIndexerPoolHost(DeepSeekV4PagedHostPool):
    """Page-row mirror of QSA compressed keys, indexed by full-KV token slots."""

    def __init__(
        self,
        device_pool: QSATokenToKVPool,
        anchor_host,
        *,
        allocator_type: str = "default",
        mtp_draft_device_pools: tuple[QSATokenToKVPool, ...] = (),
    ):
        page_size = anchor_host.page_size
        ratio = device_pool.qsa_compress_ratio
        if page_size % ratio:
            raise ValueError(
                "QSA HiCache pages must be divisible by the compress ratio"
            )
        self.device_pool = device_pool
        pools = (device_pool, *mtp_draft_device_pools)
        buffers = [b for pool in pools for b in pool.qsa_compressed_k_buffer_pool]
        if not buffers:
            raise ValueError("QSA HiCache requires at least one local indexer layer")
        item_bytes = buffers[0][0].nbytes * (page_size // ratio)
        if any(
            pool.qsa_compress_ratio != ratio
            or any(
                b[0].nbytes * (page_size // ratio) != item_bytes
                for b in pool.qsa_compressed_k_buffer_pool
            )
            for pool in pools
        ):
            raise ValueError("Packed QSA indexer pools must share the same page shape")
        # A full-KV page and its compressed groups share ownership. Packing the
        # groups into a byte row lets the existing whole-page kernels move both
        # coordinate spaces using the same full-KV page number.
        super().__init__(
            pool_name="qsa_indexer",
            device_buffers=[
                b.view(torch.uint8).reshape(-1, item_bytes) for b in buffers
            ],
            item_bytes=item_bytes,
            num_host_pages=anchor_host.page_num,
            slot_page_size=page_size,
            layout=anchor_host.layout,
            device=anchor_host.device,
            pin_memory=anchor_host.pin_memory,
            allocator_type=allocator_type,
            page_aligned_only=True,
        )
        self.size_per_token = self.get_size_per_token()

    def get_size_per_token(self):
        return self.layer_num * self.item_bytes // self.slot_page_size

    def get_ksize_per_token(self):
        return self.get_size_per_token()
