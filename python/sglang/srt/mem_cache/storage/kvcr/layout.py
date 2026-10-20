"""Translate device-pool pages into KVCR's named, registered pieces."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolGroup

if TYPE_CHECKING:
    from kvcr.types import MemoryRef, RegionDescriptor


@dataclass(frozen=True)
class PreparedTransfer:
    name: PoolName
    keys: tuple[bytes, ...]
    locations: tuple[int, ...]


class KVCRLayout:
    """One key per physical pool/page, with a named piece for each tensor."""

    def __init__(
        self,
        group: DevicePoolGroup,
        *,
        model_name: str,
        endpoint_name: str,
        tp_rank: int = 0,
        tp_size: int = 1,
        cp_rank: int = 0,
        cp_size: int = 1,
        pp_rank: int = 0,
        pp_size: int = 1,
    ):
        from kvcr.types import RegionDescriptor

        self.group = group
        self.endpoint_name = endpoint_name
        self.registrations: list[RegionDescriptor] = []
        self.pool_counts: Counter[str] = Counter()
        self.trailing_counts: Counter[str] = Counter()
        self._labels: dict[PoolName, tuple[tuple[str, ...], ...]] = {}
        pool_sizes = {}
        geometry = []
        for entry in sorted(group.entries, key=lambda entry: str(entry.name)):
            labels = []
            components = []
            for component_index, component in enumerate(entry.components):
                component_labels = []
                component_geometry = []
                for buffer_index, buffer in enumerate(component):
                    address, row_stride, size = entry.buffer_meta[component_index][
                        buffer_index
                    ]
                    row_span = entry._row_span
                    if (
                        not buffer[0].is_contiguous()
                        or size <= 0
                        or (row_span > 1 and row_stride * row_span != size)
                    ):
                        raise ValueError(
                            f"KVCR pool {entry.name} requires contiguous page pieces."
                        )
                    # KVCR allocates one fixed-size slot per named piece.
                    pool = f"{entry.name}_{size}"
                    label = f"{pool}:c{component_index}.b{buffer_index}"
                    pool_sizes[pool] = size
                    self.pool_counts[pool] += 1
                    if entry.indices_from_pool == PoolName.SWA:
                        self.trailing_counts[pool] += 1
                    self.registrations.append(
                        RegionDescriptor(
                            mem_type="VRAM",
                            device_Id=buffer.device.index or 0,
                            addr=address,
                            stride=row_stride * row_span,
                            count=buffer.shape[0] // row_span,
                            size=size,
                            label=label,
                        )
                    )
                    component_labels.append(label)
                    component_geometry.append(
                        (str(buffer.dtype), tuple(buffer.shape[1:]), size)
                    )
                labels.append(tuple(component_labels))
                components.append(component_geometry)
            self._labels[entry.name] = tuple(labels)
            geometry.append(
                (
                    str(entry.name),
                    str(entry.indices_from_pool),
                    entry.page_size,
                    entry._row_span,
                    str(getattr(entry.device_pool, "dtype", None)),
                    sorted(entry.layer_mapping.items()),
                    components,
                )
            )
        self.pool_layouts = list(pool_sizes.items())
        namespace = (
            "sglang-kvcr-v1",
            model_name,
            group.page_size,
            group.num_layers,
            None if group.rank_replicated else (tp_rank, tp_size),
            (cp_rank, cp_size),
            (pp_rank, pp_size),
            geometry,
        )
        self.namespace = hashlib.sha256(
            json.dumps(namespace, separators=(",", ":")).encode()
        ).hexdigest()
        self._key_prefix = f"sglang-kvcr-v1:{self.namespace}:".encode()

    def encode_key(self, storage_hash: str, pool_name: PoolName) -> bytes:
        return self._key_prefix + f"{pool_name}:{storage_hash}".encode()

    def decode_key(self, key: bytes) -> tuple[PoolName, str] | None:
        if not key.startswith(self._key_prefix):
            return None
        try:
            name, storage_hash = key[len(self._key_prefix) :].decode().split(":", 1)
            pool_name = PoolName(name)
        except (ValueError, UnicodeDecodeError):
            return None
        if pool_name not in self.group.entry_map:
            return None
        return pool_name, storage_hash

    def encoded_keys(
        self, transfers: list[PoolTransfer], *, allow_partial: bool = False
    ) -> set[bytes]:
        """Storage keys of a transfer set without reading device indices."""
        return {
            self.encode_key(key, transfer.name)
            for transfer in self.group.resolve_transfers(
                transfers, allow_partial=allow_partial
            )
            for key in transfer.keys
        }

    def prepare(
        self,
        transfers: list[PoolTransfer],
        *,
        allow_partial: bool = False,
        allow_missing_kv: bool = False,
    ) -> tuple[PreparedTransfer, ...]:
        """Resolve and snapshot indices once; queries may omit device indices."""
        resolved = self.group.resolve_transfers(
            transfers,
            allow_partial=allow_partial,
            allow_missing_kv=allow_missing_kv,
        )
        prepared = []
        for transfer in resolved:
            storage_keys = tuple(transfer.keys)
            locations = ()
            if transfer.host_indices is not None:
                locations = tuple(
                    self.group.entry_map[transfer.name].prepare_locations(
                        transfer.host_indices
                    )
                )
                if len(storage_keys) != len(locations):
                    raise ValueError(
                        f"KVCR pool {transfer.name} has {len(storage_keys)} keys "
                        f"but {len(locations)} pages."
                    )
            prepared.append(
                PreparedTransfer(
                    name=transfer.name,
                    keys=tuple(
                        self.encode_key(key, transfer.name) for key in storage_keys
                    ),
                    locations=locations,
                )
            )
        return tuple(prepared)

    def mappings(
        self, prepared: Sequence[PreparedTransfer], layer: int | None = None
    ) -> list[dict[bytes, list[MemoryRef]]]:
        """Batch pools together while preserving repeated keys' destinations."""
        from kvcr.types import MemoryRef

        batches: list[dict[bytes, list[MemoryRef]]] = []
        occurrences: Counter[bytes] = Counter()
        for transfer in prepared:
            if len(transfer.keys) != len(transfer.locations):
                raise ValueError(
                    f"KVCR pool {transfer.name} transfer has no locations."
                )
            entry = self.group.entry_map[transfer.name]
            components = self._labels[transfer.name]
            if layer is None:
                labels = [label for component in components for label in component]
            else:
                mapped = entry.layer_mapping.get(layer)
                if mapped is None:
                    continue
                indices = (mapped,) if isinstance(mapped, int) else mapped
                labels = [
                    component[index] for component in components for index in indices
                ]
            for key, location in zip(transfer.keys, transfer.locations):
                batch_index = occurrences[key]
                occurrences[key] += 1
                if batch_index == len(batches):
                    batches.append({})
                batches[batch_index][key] = [
                    MemoryRef(
                        end_point_name=self.endpoint_name,
                        label=label,
                        element_index=location // entry._row_span,
                    )
                    for label in labels
                ]
        return batches
