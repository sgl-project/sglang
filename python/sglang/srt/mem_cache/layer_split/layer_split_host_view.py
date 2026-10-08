"""Physical DSA shard views shared by L3 prefetch and backup.

Only this module interprets the host pools' tensor layout. Transport code moves
opaque bytes through read_page/write_page and cannot index host tensors itself.
The KV anchor remains the sole allocator; INDEXER follows its token indices.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.layer_split.layer_split_utils import owned_layer_range
from sglang.srt.mem_cache.pool_host import HostPoolGroup


class LayerSplitShardView:
    def __init__(self, pool, component, *, owned_layers, capacity_layers, page_size):
        self.pool = pool
        self.component = component
        self.owned_layers = owned_layers
        self.capacity_layers = capacity_layers
        self.page_size = page_size
        if component == "target":
            self.row_bytes = int(pool.token_stride_size)
        elif component == "indexer":
            self.row_bytes = int(pool.indexer_page_stride_size) // page_size
        else:
            raise ValueError(f"Unknown DSA component: {component}")
        self.validate()

    def validate(self):
        pool = self.pool
        if pool.layout != "layer_first":
            raise ValueError("LayerSplit host views require layer_first layout")
        if getattr(pool, "mtp_draft_device_pools", ()) or (
            getattr(pool, "target_layer_num", None) is not None
            and pool.layer_num != pool.target_layer_num
        ):
            raise ValueError("LayerSplit storage does not support draft layers")
        if self.page_size <= 0 or self.row_bytes <= 0:
            raise ValueError("Page size and component row width must be positive")
        if not 0 <= self.owned_layers <= self.capacity_layers:
            raise ValueError("Owned layers exceed host capacity")
        if self.component == "target":
            buffer = pool.kv_buffer
            if buffer.ndim != 4 or buffer.shape[2] != 1:
                raise ValueError("KV host buffer must be [layers, tokens, 1, dim]")
            width = buffer.shape[-1] * buffer.element_size()
        else:
            buffer = pool.index_k_with_scale_buffer
            if buffer.ndim != 3 or buffer.dtype != torch.uint8:
                raise ValueError(
                    "Indexer host buffer must be uint8 [layers, pages, stride]"
                )
            width = buffer.shape[-1]
        expected_width = self.row_bytes * (
            self.page_size if self.component == "indexer" else 1
        )
        if width != expected_width or buffer.shape[0] != self.capacity_layers:
            raise ValueError("Host buffer shape differs from the DSA shard geometry")
        if buffer.device.type != "cpu" or not buffer.is_contiguous():
            raise ValueError("LayerSplit L2 storage must be contiguous CPU memory")

    def _page(self, token_base):
        if token_base < 0 or token_base % self.page_size:
            raise ValueError("Host token index must be nonnegative and page-aligned")
        if self.component == "target":
            buffer = self.pool.kv_buffer.view(torch.uint8)
            if token_base + self.page_size > buffer.shape[1]:
                raise ValueError("Host page exceeds KV capacity")
            return buffer[
                : self.owned_layers, token_base : token_base + self.page_size, 0, :
            ]
        page = token_base // self.page_size
        buffer = self.pool.index_k_with_scale_buffer
        if page >= buffer.shape[1]:
            raise ValueError("Host page exceeds indexer capacity")
        return buffer[: self.owned_layers, page, :].view(
            self.owned_layers, self.page_size, self.row_bytes
        )

    def _check_payload(self, payload):
        if payload.dtype != torch.uint8 or payload.device.type != "cpu":
            raise ValueError("Shard payload must be CPU uint8 bytes")
        if payload.numel() != self.owned_layers * self.page_size * self.row_bytes:
            raise ValueError("Shard payload has the wrong byte count")

    def read_page(self, token_base, destination):
        self._check_payload(destination)
        destination.view(self.owned_layers, self.page_size, self.row_bytes).copy_(
            self._page(token_base)
        )

    def write_page(self, token_base, source):
        self._check_payload(source)
        self._page(token_base).copy_(
            source.view(self.owned_layers, self.page_size, self.row_bytes)
        )


class LayerSplitHostView:
    """Non-owning shard access over the native KV anchor and indexer sidecar."""

    def __init__(self, host_pool_group: HostPoolGroup):
        if not isinstance(host_pool_group, HostPoolGroup):
            raise TypeError("LayerSplitHostView requires the native HostPoolGroup")
        self.host_pool_group = host_pool_group
        self.page_size = host_pool_group.page_size
        if set(host_pool_group.entry_map) != {PoolName.KV, PoolName.INDEXER}:
            raise ValueError("LayerSplit storage requires exactly KV and INDEXER")
        device_pool = host_pool_group.entry_map[PoolName.KV].device_pool
        if not device_pool.layer_shard_enabled:
            raise ValueError(
                "LayerSplitHostView requires a layer-sharded DSA device pool"
            )
        self.layer_count = device_pool.layer_num
        self.shard_size = device_pool.layer_shard_size
        self.shard_rank = device_pool.layer_shard_rank
        start, end = device_pool._owned_local_layer_range()
        self.owned_layers = end - start
        # Index-share models only allocate the layers declared by the indexer
        # pool. Its compact shard can have fewer layers than the KV shard.
        declarations = {decl.pool_name: decl for decl in device_pool.host_pool_decls()}
        ranges = [
            owned_layer_range(rank, self.shard_size, self.layer_count)
            for rank in range(self.shard_size)
        ]
        self.component_layer_counts = {}
        for component, name in (("target", PoolName.KV), ("indexer", PoolName.INDEXER)):
            declared = declarations[name].owned_device_layers
            layers = range(self.layer_count) if declared is None else declared
            self.component_layer_counts[component] = tuple(
                sum(lo <= layer < hi for layer in layers) for lo, hi in ranges
            )
        self.shard_views = {
            component: LayerSplitShardView(
                host_pool_group.get_pool(name),
                component,
                owned_layers=self.component_layer_counts[component][self.shard_rank],
                capacity_layers=host_pool_group.get_pool(name).layer_num,
                page_size=self.page_size,
            )
            for component, name in (
                ("target", PoolName.KV),
                ("indexer", PoolName.INDEXER),
            )
        }
