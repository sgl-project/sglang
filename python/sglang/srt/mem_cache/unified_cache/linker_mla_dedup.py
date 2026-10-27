"""Backend-independent MLA broadcast adapter for external cache linkers."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.mla_host_dedup import MLAHostDedupBroadcaster
from sglang.srt.utils import get_device_module

device_module = get_device_module()


class LinkerMLADedupBroadcaster(MLAHostDedupBroadcaster):
    """Adapt hybrid pool geometry to HiCache's existing broadcast primitive.

    Native MLA/DSA pools can use MLAHostDedupBroadcaster directly. This adapter
    covers multiple physical pools and sparse layers (e.g. DSV4). The caller
    must provide the same pools and logical page order on all ranks; only
    physical slots may differ. Call on the forward stream after source KV is
    loaded. Read completion, failure coordination and scheduling stay outside.
    """

    def __init__(self, pool_group, group, src_global_rank):
        if not pool_group.rank_replicated:
            raise ValueError("Linker broadcasts require rank-replicated pools.")
        self.pools = pool_group.entry_map
        self.buffers = {
            name: [
                [buf.view(torch.uint8).view(buf.shape[0], -1) for buf in component]
                for component in pool.components
            ]
            for name, pool in self.pools.items()
        }
        first = next(iter(self.buffers.values()))[0][0]
        # Reuse HiCache's allocation and token-based chunk budget, even when
        # the physical buffers store page rows rather than individual tokens.
        bytes_per_token = max(
            (size + pool.page_size - 1) // pool.page_size
            for pool in self.pools.values()
            for component in pool.buffer_meta
            for _, _, size in component
        )
        super().__init__(
            SimpleNamespace(
                device=first.device,
                layer_num=pool_group.num_layers,
                kv_cache_dim=bytes_per_token,
                kv_buffer=[first],
            ),
            group,
            src_global_rank,
        )
        # A physical page cannot be split into smaller rows by _bcast_layer.
        max_row_bytes = max(
            buf.shape[1]
            for components in self.buffers.values()
            for component in components
            for buf in component
        )
        self.kv_staging.resize_(max(self.kv_staging.numel(), max_row_bytes))

    def prepare_broadcast(self, indices_by_pool, load_stream):
        """Map already-translated device slots to each physical pool's rows."""
        prepared = {}
        for name, indices in indices_by_pool.items():
            pool = self.pools[name]
            rows = torch.tensor(pool.prepare_locations(indices), dtype=torch.int64)
            rows = (rows[:, None] + torch.arange(pool._row_span)).flatten()
            prepared[name] = super().prepare_broadcast(rows, load_stream)
        return prepared

    def broadcast_loaded_layer(self, layer_id, prepared):
        for name, pool in self.pools.items():
            layer = pool.layer_mapping.get(layer_id)
            if layer is None or name not in prepared:
                continue
            indices, _ = prepared[name]
            if indices.is_cuda:
                indices.record_stream(device_module.current_stream())
            for buffers in self.buffers[name]:
                self._bcast_layer(
                    buffers, self.kv_staging, indices, buffers[layer].shape[1], layer
                )


class LinkerLoadBroadcast:
    """Broadcast each layer the source rank loads to its rank-replicated peers."""

    def __init__(self, broadcaster: LinkerMLADedupBroadcaster, tp_group, num_layers):
        self.broadcaster = broadcaster
        self.tp_group = tp_group
        self.num_layers = num_layers
        # counter index -> (next layer to broadcast, prepared indices, ready event)
        self.batches = {}

    @classmethod
    def build(cls, pool_group, tp_cache_group, attn_tp_cache_group, tp_group):
        broadcaster = LinkerMLADedupBroadcaster.build(
            pool_group, tp_cache_group, attn_tp_cache_group
        )
        return cls(broadcaster, tp_group, pool_group.num_layers)

    @property
    def is_src(self) -> bool:
        return self.broadcaster.is_src

    def plans_agree(self, pending) -> bool:
        """Collective, run every batch: compare logical load order, not local slots."""
        plan = [
            (rid, [(t.name, list(t.keys)) for t in transfers])
            for rid, transfers in pending.items()
        ]
        digest = hashlib.sha256(repr(plan).encode()).digest()
        digests = [None] * torch.distributed.get_world_size(self.tp_group)
        torch.distributed.all_gather_object(digests, digest, group=self.tp_group)
        return all(other == digest for other in digests)

    def begin(self, index: int, pending) -> None:
        indices = {}
        for transfers in pending.values():
            for transfer in transfers:
                indices.setdefault(transfer.name, []).append(transfer.host_indices)
        prepared = self.broadcaster.prepare_broadcast(
            {name: torch.cat(parts) for name, parts in indices.items()},
            device_module.current_stream(),
        )
        ready_event = device_module.Event()
        ready_event.record()
        self.batches[index] = (0, prepared, ready_event)

    def broadcast_until(self, index: int, threshold: int) -> None:
        batch = self.batches.get(index)
        if batch is None:
            return
        first, prepared, ready_event = batch
        if first == 0:
            device_module.current_stream().wait_event(ready_event)
        # KV access may wait several times per layer, or skip sparse layers.
        # Launch each broadcast exactly once, in order, on the forward stream.
        for layer in range(first, threshold + 1):
            self.broadcaster.broadcast_loaded_layer(layer, prepared)
        if threshold == self.num_layers - 1:
            del self.batches[index]
        else:
            self.batches[index] = (max(first, threshold + 1), prepared, ready_event)

    def reset(self) -> None:
        self.batches.clear()

    def destroy(self) -> None:
        self.broadcaster.destroy()
