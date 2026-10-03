"""Gather selected target KV into request-owned Host or device storage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import torch
from sglang.srt.training_capture.protocol import DTYPES, ContractError, KVSpec
from sglang.srt.training_capture.topology import CapturePartition


class SelectedLayerKVExporter:
    def __init__(
        self,
        kv: KVSpec,
        buffers: Mapping[str, torch.Tensor],
        *,
        partition: CapturePartition | None = None,
    ):
        self.kv = kv
        self.buffers = dict(buffers)
        if not self.buffers:
            raise ContractError("KV exporter requires selected source buffers")
        self.device = next(iter(self.buffers.values())).device
        expected = set()
        layers = kv.layers if partition is None else partition.local_layers(kv)
        for layer in layers:
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            ):
                name = f"target_{component}.{layer.layer_id}"
                expected.add(name)
                value = self.buffers[name]
                if value.device != self.device:
                    raise ContractError("selected KV buffers must share a device")
                if (
                    value.ndim != 3
                    or list(value.shape[1:]) != [layer.num_kv_heads, dim]
                    or value.dtype != DTYPES[kv.dtype]
                ):
                    raise ContractError(
                        "KV exporter requires unquantized token/head/dim source buffers"
                    )
        if set(self.buffers) != expected:
            raise ContractError("KV source layers differ from the capture contract")

    @classmethod
    def from_pool(cls, kv: KVSpec, pool, *, partition: CapturePartition | None = None):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if (
            not isinstance(pool, MHATokenToKVPool)
            or pool.is_quantized_kv_cache
            or pool.use_hnd
        ):
            raise ContractError(
                "capture requires a dense, unquantized NHD MHA/GQA pool"
            )
        if pool.page_size != kv.source_page_size:
            raise ContractError("configured source page size disagrees with the pool")
        buffers = {}
        layers = kv.layers if partition is None else partition.local_layers(kv)
        for layer in layers:
            if not 0 <= layer.layer_id - pool.start_layer < pool.layer_num:
                raise ContractError("selected layer is not local to this worker")
            buffers[f"target_k.{layer.layer_id}"] = pool.get_key_buffer(layer.layer_id)
            buffers[f"target_v.{layer.layer_id}"] = pool.get_value_buffer(
                layer.layer_id
            )
        return cls(kv, buffers, partition=partition)

    def bind(self, slot):
        """Bind immutable, pool-owned destinations before capture admission."""
        return BoundHiCacheKVExporter(self, slot)

    @torch.no_grad()
    def export(
        self,
        slots: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        start: int,
        end: int,
    ) -> None:
        """Enqueue on the producer stream, before any reuse of the source slots.

        Gathered temporaries own storage. Same-stream ordering protects the
        pool read, and record_stream protects temporary storage through D2H.
        The caller records completion and retains Host storage until it fires.
        """
        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.numel() != end - start
            or not 0 <= start < end
        ):
            raise ContractError("invalid KV export position mapping")
        # index_select accepts the request pool's int32 indices without an upcast.
        indices = slots.to(device=self.device)
        stream = torch.cuda.current_stream(self.device) if indices.is_cuda else None
        if stream is not None:
            indices.record_stream(stream)
        for name, source in self.buffers.items():
            destination = destinations[name][start:end]
            if (
                destination.shape != (end - start, *source.shape[1:])
                or destination.dtype != source.dtype
            ):
                raise ContractError("destination does not cover the KV range")
            if destination.is_cuda:
                if destination.device != source.device:
                    raise ContractError("KV staging must share the source device")
                torch.index_select(source, 0, indices, out=destination)
                destination.record_stream(stream)
                continue
            if destination.device.type != "cpu":
                raise ContractError("KV export requires Host or CUDA storage")
            if source.is_cuda and not destination.is_pinned():
                raise ContractError(
                    "asynchronous KV export requires pinned Host storage"
                )
            gathered = source.index_select(0, indices)
            destination.copy_(gathered, non_blocking=source.is_cuda)
            if stream is not None:
                gathered.record_stream(stream)


class BoundHiCacheKVExporter:
    """Use existing HiCache kernels with metadata owned by the registered slot."""

    def __init__(self, exporter, slot):
        from sglang.kernels.ops.kvcache.hicache import can_use_hicache_jit_kernel

        self.exporter = exporter
        self.metadata = slot.kv_export_tensors
        self.destinations = (slot.tensors, slot.device_tensors)
        self.host_storage, self.device_storage = slot.storage, slot.device_storage
        self.sources = tuple(exporter.buffers.values())
        self.host_enqueued_bytes = self.device_enqueued_bytes = 0
        if self.metadata is None or exporter.device.type != "cuda":
            raise ContractError(
                "HiCache export metadata must be allocated by the Host pool"
            )
        groups = defaultdict(list)
        for name, source in exporter.buffers.items():
            row_bytes = source.shape[1] * source.shape[2] * source.element_size()
            if (
                source.stride(2) != 1
                or source.stride(1) != source.shape[2]
                or row_bytes % 128
                or source.data_ptr() % 16
                or source.stride(0) * source.element_size() % 16
            ):
                raise ContractError(
                    "HiCache export requires aligned, contiguous KV rows of a multiple of 128 bytes"
                )
            for tensors in self.destinations:
                if tensors is None:
                    continue
                destination = tensors[name]
                if (
                    destination.shape[1:] != source.shape[1:]
                    or destination.dtype != source.dtype
                    or not destination.is_contiguous()
                    or destination.data_ptr() % 16
                    or (
                        destination.device != source.device
                        and not (
                            destination.device.type == "cpu" and destination.is_pinned()
                        )
                    )
                ):
                    raise ContractError(
                        "HiCache export destination differs from the bound KV layout"
                    )
            groups[row_bytes, source.stride(0) * source.element_size()].append(name)
        self.groups = []
        names = []
        for (row_bytes, source_stride), group_names in groups.items():
            if not can_use_hicache_jit_kernel(element_size=row_bytes):
                raise ContractError("HiCache KV export kernel could not be initialized")
            self.groups.append(
                (len(names), len(names) + len(group_names), row_bytes, source_stride)
            )
            names.extend(group_names)
        self.source_rows = min(value.shape[0] for value in exporter.buffers.values())
        self.row_bytes = sum(
            (last - first) * width for first, last, width, _ in self.groups
        )
        self.metadata["export_sources"].copy_(
            torch.tensor(
                [exporter.buffers[name].data_ptr() for name in names],
                dtype=torch.uint64,
            )
        )
        for key, tensors in zip(("export_host", "export_staging"), self.destinations):
            if tensors is not None:
                self.metadata[key].copy_(
                    torch.tensor(
                        [tensors[name].data_ptr() for name in names], dtype=torch.uint64
                    )
                )
        for bits in (32, 64):
            positions = self.metadata[f"export_positions{bits}"]
            torch.arange(positions.numel(), out=positions)

    @torch.no_grad()
    def export(self, slots, destinations, start, end):
        from sglang.kernels.ops.kvcache.hicache import transfer_hicache_all_layer_mla

        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.numel() != end - start
            or not 0 <= start < end
        ):
            raise ContractError("invalid KV export position mapping")
        if destinations is self.destinations[0]:
            pointer_key = "export_host"
        elif destinations is self.destinations[1] and destinations is not None:
            pointer_key = "export_staging"
        else:
            raise ContractError("KV destinations were not bound to this capture slot")
        if any(end > destinations[name].shape[0] for name in self.exporter.buffers):
            raise ContractError("destination does not cover the KV range")
        indices = slots.to(device=self.exporter.device).contiguous()
        # The pointer kernel has no pool extent parameter. Keep asynchronous
        # bounds rejection equivalent to index_select before it accesses storage.
        torch._assert_async(
            ((indices >= 0) & (indices < self.source_rows)).all(),
            "capture KV source index out of range",
        )
        stream = torch.cuda.current_stream(self.exporter.device)
        indices.record_stream(stream)
        bits = 32 if indices.dtype == torch.int32 else 64
        positions = self.metadata[f"export_positions{bits}"][start:end]
        for first, last, row_bytes, source_stride in self.groups:
            transfer_hicache_all_layer_mla(
                ptr_dst=self.metadata[pointer_key][first:last],
                indices_dst=positions,
                ptr_src=self.metadata["export_sources"][first:last],
                indices_src=indices,
                cache_src_stride_bytes=source_stride,
                cache_dst_stride_bytes=row_bytes,
                element_size=row_bytes,
            )
        self.device_storage.record_stream(stream)
        if pointer_key == "export_host":
            self.host_enqueued_bytes += (end - start) * self.row_bytes
        else:
            self.device_enqueued_bytes += (end - start) * self.row_bytes
