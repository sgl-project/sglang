"""Host pool for QSA compressed index keys.

Compressed keys hold one group per ``compress_ratio`` full-KV slots at
``full_slot // ratio``, so a full-KV page owns a fixed run of groups. The host
pool moves that run as one byte row per page with the whole-page kernels of
DeepSeekV4PagedHostPool. The per-request pending ring (index keys of the group
still being filled) is transient per-request data and is not backed up; a host
restore is page aligned, so the restored prefix ends on a group boundary and
the ring starts empty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.host_pool_decl import (
    HostPoolDecl,
    HostPoolStorageInfo,
)

if TYPE_CHECKING:
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


class QSAIndexerHostPoolBuilder:
    def validate(
        self,
        *,
        decl: HostPoolDecl,
        page_size: int,
        packed_draft_device_pools: tuple[QSATokenToKVPool, ...],
    ) -> None:
        _qsa_device_page_buffers(
            decl=decl,
            page_size=page_size,
            packed_draft_device_pools=packed_draft_device_pools,
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
            decl=decl,
            anchor_host=anchor_host,
            packed_draft_device_pools=packed_draft_device_pools,
            allocator_type=allocator_type,
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


def _qsa_device_page_buffers(
    *,
    decl: HostPoolDecl,
    page_size: int,
    packed_draft_device_pools: tuple[QSATokenToKVPool, ...],
) -> list[torch.Tensor]:
    item_bytes = decl.storage_info.page_bytes(page_size)
    rows = []
    for pool in (decl.device_pool, *packed_draft_device_pools):
        if page_size % pool.qsa_compress_ratio:
            raise ValueError(
                f"HiCache page {page_size} is not a multiple of the QSA "
                f"compress ratio {pool.qsa_compress_ratio}"
            )
        groups_per_page = page_size // pool.qsa_compress_ratio
        for buffer in pool.qsa_compressed_k_buffer_pool:
            if buffer.dtype != decl.storage_info.dtype or not buffer.is_contiguous():
                raise ValueError(
                    f"{decl.pool_name}: compressed keys require matching dtype and contiguous storage"
                )
            page_bytes = buffer[0].nbytes * groups_per_page
            if page_bytes != item_bytes:
                raise ValueError(
                    f"{type(pool).__name__} compressed keys take {page_bytes} "
                    f"bytes per page, declared {item_bytes}"
                )
            rows.append(buffer.view(torch.uint8).reshape(-1, item_bytes))
    if not rows:
        raise ValueError(f"{decl.pool_name} declared with no compressed key layers")
    return rows


class QSAIndexerPoolHost(DeepSeekV4PagedHostPool):
    """Page rows of compressed keys addressed by full-KV token slots; packed
    drafts append their layers after the target's."""

    def __init__(
        self,
        *,
        decl: HostPoolDecl,
        anchor_host: Any,
        packed_draft_device_pools: tuple[QSATokenToKVPool, ...] = (),
        allocator_type: str = "default",
    ):
        self.decl = decl
        self.device_pool = decl.device_pool
        page_size = anchor_host.page_size
        item_bytes = decl.storage_info.page_bytes(page_size)
        rows = _qsa_device_page_buffers(
            decl=decl,
            page_size=page_size,
            packed_draft_device_pools=packed_draft_device_pools,
        )
        super().__init__(
            pool_name=decl.pool_name.value,
            device_buffers=rows,
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
