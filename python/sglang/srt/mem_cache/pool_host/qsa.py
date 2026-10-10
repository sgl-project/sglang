from __future__ import annotations

from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.host_pool_decl import (
    HostPoolDecl,
    HostPoolStorageInfo,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


def qsa_indexer_bytes_per_token_per_layer(
    *, kv_heads: int, head_dim: int, compress_ratio: int, dtype: torch.dtype
) -> int:
    group_bytes = kv_heads * head_dim * dtype.itemsize
    if group_bytes % compress_ratio:
        raise ValueError(
            f"QSA compressed group of {group_bytes} bytes is not a whole number "
            f"of bytes per token at compress ratio {compress_ratio}"
        )
    return group_bytes // compress_ratio


def _qsa_page_item_bytes(pools: tuple[QSATokenToKVPool, ...], page_size: int) -> int:
    """Bytes of one page row of compressed keys, shared by every packed pool."""
    ratio = pools[0].qsa_compress_ratio
    if page_size % ratio:
        raise ValueError("QSA HiCache pages must be divisible by the compress ratio")
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
    return item_bytes


class QSAIndexerHostPoolBuilder:
    def validate(
        self,
        *,
        decl: HostPoolDecl,
        transfer_page_size: int,
        packed_draft_device_pools: tuple[QSATokenToKVPool, ...],
    ) -> None:
        item_bytes = _qsa_page_item_bytes(
            (decl.device_pool, *packed_draft_device_pools), transfer_page_size
        )
        declared = decl.storage_info.page_bytes(transfer_page_size)
        if item_bytes != declared:
            raise ValueError(
                f"{decl.pool_name}: compressed keys take {item_bytes} bytes per "
                f"page, declared {declared}"
            )

    def build(
        self,
        *,
        decl: HostPoolDecl,
        anchor_host: Any,
        allocator_type: str,
        packed_draft_device_pools: tuple[QSATokenToKVPool, ...],
    ) -> QSAIndexerPoolHost:
        return QSAIndexerPoolHost(
            decl.device_pool,
            anchor_host,
            allocator_type=allocator_type,
            mtp_draft_device_pools=packed_draft_device_pools,
        )

    def kv_budget_bytes(
        self,
        *,
        decl: HostPoolDecl,
        packed_draft_device_pools: tuple[QSATokenToKVPool, ...],
    ) -> int:
        return sum(
            b.nbytes
            for pool in (decl.device_pool, *packed_draft_device_pools)
            for b in pool.qsa_compressed_k_buffer_pool
        )


def make_qsa_indexer_pool_decl(
    pool: QSATokenToKVPool, *, name: PoolName = PoolName.INDEXER
) -> HostPoolDecl:
    """Compressed keys riding on the full-KV pages: indices and layout both follow KV."""
    return HostPoolDecl(
        pool_name=name,
        device_pool=pool,
        indices_from_pool=PoolName.KV,
        layout_source=PoolName.KV,
        storage_info=HostPoolStorageInfo(
            bytes_per_token_per_layer=qsa_indexer_bytes_per_token_per_layer(
                kv_heads=pool.qsa_index_kv_heads,
                head_dim=pool.qsa_index_head_dim,
                compress_ratio=pool.qsa_compress_ratio,
                dtype=pool.index_state_dtype,
            ),
            dtype=pool.index_state_dtype,
        ),
        host_pool_builder=QSAIndexerHostPoolBuilder(),
    )


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
        self.device_pool = device_pool
        pools = (device_pool, *mtp_draft_device_pools)
        item_bytes = _qsa_page_item_bytes(pools, page_size)
        buffers = [b for pool in pools for b in pool.qsa_compressed_k_buffer_pool]
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
