"""Thin SGLang adapter for Mooncake's KV-cache manifest planner.

SGLang owns request routing, page lifetime, ZMQ connections, and completion
state.  This module owns only translation to Mooncake contracts, topology
planning, request-time address binding, and TE submission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np

KV_RESHARD_PROTOCOL = "KV_RESHARD_V1"
KV_RESHARD_SCHEMA_VERSION = 1


class KVReshardCompatibilityError(RuntimeError):
    pass


def _weight_generation(weight_version: str) -> int:
    digest = hashlib.sha256(weight_version.encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _head_placement(
    total_kv_heads: int, tp_rank: int, tp_size: int
) -> tuple[int, int, int, int]:
    if total_kv_heads >= tp_size and total_kv_heads % tp_size == 0:
        head_count = total_kv_heads // tp_size
        return tp_rank * head_count, head_count, 0, 1
    if tp_size > total_kv_heads and tp_size % total_kv_heads == 0:
        replica_count = tp_size // total_kv_heads
        return tp_rank // replica_count, 1, tp_rank % replica_count, replica_count
    raise KVReshardCompatibilityError(
        "Mooncake KV reshard requires TP and total KV heads to have an exact "
        f"shard/replica ratio, got total_kv_heads={total_kv_heads}, tp_size={tp_size}"
    )


def _participant_id(role: str, dp_rank: int, pp_rank: int, tp_rank: int) -> str:
    return f"{role}:dp{dp_rank}:pp{pp_rank}:tp{tp_rank}"


@dataclass(frozen=True)
class KVReshardRoutePlan:
    bootstrap_infos: tuple[dict[str, Any], ...]
    expected_writer_ids: tuple[str, ...]


@dataclass(frozen=True)
class KVReshardNativeBatch:
    endpoint: str
    source_addresses: tuple[int, ...]
    target_addresses: tuple[int, ...]
    sizes: tuple[int, ...]


class KVReshardRuntime:
    """One rank's address-free placement and optional physical binding."""

    def __init__(
        self,
        *,
        kv_args: Any,
        server_args: Any,
        role: str,
        dp_rank: int,
        dp_size: int,
        pp_rank: int,
        pp_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        from mooncake.reshard.kv_cache import KVCachePlacementPart

        if role not in ("prefill", "decode"):
            raise ValueError(f"invalid KV reshard role: {role}")
        self.kv_args = kv_args
        self.role = role
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.total_layers = int(getattr(kv_args, "total_kv_layers", 0))
        self.total_kv_heads = int(getattr(kv_args, "total_kv_head_num", 0))
        self.head_count = int(getattr(kv_args, "kv_head_num", 0))
        self.key_head_dim = int(getattr(kv_args, "kv_head_dim", 0))
        self.value_head_dim = int(getattr(kv_args, "kv_value_head_dim", 0))
        self.itemsize = int(getattr(kv_args, "kv_itemsize", 0))
        self.dtype = str(getattr(kv_args, "kv_storage_dtype_str", ""))
        self.page_size = int(kv_args.page_size)
        self.layer_ids = self._resolve_layer_ids()
        self.revision = (
            f"{server_args.model_path}@{server_args.revision or 'default'}"
            f"#{server_args.weight_version}"
        )
        self.weight_generation = _weight_generation(server_args.weight_version)
        self.resource_id = f"kv:{server_args.model_path}"
        self.instance_id = f"{role}:dp{dp_rank}"
        self.participant_id = _participant_id(role, dp_rank, pp_rank, tp_rank)
        self._validate_geometry()
        head_start, head_count, replica_ordinal, replica_count = _head_placement(
            self.total_kv_heads, tp_rank, tp_size
        )
        if head_count != self.head_count:
            raise KVReshardCompatibilityError(
                "SGLang local KV head count differs from the declared TP placement: "
                f"runtime={self.head_count}, planned={head_count}"
            )
        self.local_part = KVCachePlacementPart(
            resource_id=self.resource_id,
            revision=self.revision,
            weight_generation=self.weight_generation,
            participant_id=self.participant_id,
            instance_id=self.instance_id,
            dp_rank=dp_rank,
            dp_size=dp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            layer_ids=self.layer_ids,
            total_kv_heads=self.total_kv_heads,
            head_start=head_start,
            head_count=head_count,
            replica_ordinal=replica_ordinal,
            replica_count=replica_count,
            dtype=self.dtype,
            itemsize=self.itemsize,
            page_size=self.page_size,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
        )
        self.binding = None
        self.transfer_engine = None
        self._complete_placement = None

    def _resolve_layer_ids(self) -> tuple[int, ...]:
        ptr_count = len(self.kv_args.kv_data_ptrs)
        if ptr_count == 0 or ptr_count % 2:
            raise KVReshardCompatibilityError(
                "Mooncake KV reshard requires separate K/V buffers per layer"
            )
        layer_count = ptr_count // 2
        explicit = tuple(int(v) for v in (self.kv_args.kv_layer_ids or []))
        if explicit:
            if len(explicit) != layer_count:
                raise KVReshardCompatibilityError(
                    "KV layer ID count differs from the K/V buffer count"
                )
            return explicit
        start = int(getattr(self.kv_args, "prefill_start_layer", 0))
        return tuple(range(start, start + layer_count))

    def _validate_geometry(self) -> None:
        checks = {
            "total_kv_layers": self.total_layers,
            "total_kv_head_num": self.total_kv_heads,
            "kv_head_num": self.head_count,
            "kv_head_dim": self.key_head_dim,
            "kv_value_head_dim": self.value_head_dim,
            "kv_itemsize": self.itemsize,
        }
        missing = [name for name, value in checks.items() if value <= 0]
        if missing:
            raise KVReshardCompatibilityError(
                f"Mooncake KV reshard is missing MHA/GQA geometry: {missing}"
            )
        if getattr(self.kv_args, "kv_cache_layout", "nhd").lower() != "nhd":
            raise KVReshardCompatibilityError(
                "Mooncake KV reshard V1 supports only NHD KV cache layout"
            )
        if getattr(self.kv_args, "kv_is_quantized", False):
            raise KVReshardCompatibilityError(
                "Mooncake KV reshard V1 does not support quantized KV caches"
            )
        if not self.dtype:
            raise KVReshardCompatibilityError("KV storage dtype is unavailable")
        if not set(self.layer_ids).issubset(range(self.total_layers)):
            raise KVReshardCompatibilityError("KV layer IDs exceed the model layers")
        if len(self.kv_args.kv_data_ptrs) != 2 * len(self.layer_ids):
            raise KVReshardCompatibilityError(
                "Draft or non-layer-indexed KV buffers are not supported in V1"
            )

    def bind_runtime(self, *, session_id: str, transfer_engine: Any) -> None:
        from mooncake.reshard.contracts import RuntimeBindingFragment
        from mooncake.reshard.kv_cache import (
            KVCacheBufferBinding,
            KVCacheComponent,
            KVCacheRuntimeBindingManifest,
            placement_fragment_id,
        )

        layer_count = len(self.layer_ids)
        buffers = []
        for component, offset, head_dim in (
            (KVCacheComponent.KEY, 0, self.key_head_dim),
            (KVCacheComponent.VALUE, layer_count, self.value_head_dim),
        ):
            row_bytes = self.head_count * head_dim * self.itemsize
            expected_page_bytes = row_bytes * self.page_size
            for local_index, layer_id in enumerate(self.layer_ids):
                index = offset + local_index
                address = int(self.kv_args.kv_data_ptrs[index])
                nbytes = int(self.kv_args.kv_data_lens[index])
                item_len = int(self.kv_args.kv_item_lens[index])
                if item_len != expected_page_bytes or nbytes % row_bytes:
                    raise KVReshardCompatibilityError(
                        "SGLang KV buffer bytes differ from the NHD manifest geometry"
                    )
                fragment = RuntimeBindingFragment(
                    placement_fragment_id=placement_fragment_id(
                        self.participant_id, layer_id, component
                    ),
                    fragment_id=(
                        f"{self.participant_id}:{layer_id}:{component.value}:"
                        f"{self.weight_generation}"
                    ),
                    address=address,
                    nbytes=nbytes,
                    worker_id=self.participant_id,
                    endpoint=session_id,
                    device=f"cuda:{self.kv_args.gpu_id}",
                    itemsize=self.itemsize,
                    local_shape=(nbytes // row_bytes, self.head_count, head_dim),
                    strides_bytes=(row_bytes, head_dim * self.itemsize, self.itemsize),
                    storage_address=address,
                    storage_nbytes=nbytes,
                    storage_offset_bytes=0,
                )
                buffers.append(KVCacheBufferBinding(layer_id, component, fragment))
        self.binding = KVCacheRuntimeBindingManifest(
            resource_id=self.resource_id,
            placement_id=self.local_part.placement_id,
            placement_digest=self.local_part.digest,
            instance_id=self.instance_id,
            generation=self.weight_generation,
            lease_id=f"static:{self.participant_id}:{session_id}",
            revision=self.revision,
            weight_generation=self.weight_generation,
            participant_id=self.participant_id,
            buffers=tuple(buffers),
        )
        self.transfer_engine = transfer_engine

    @staticmethod
    def assemble_placement(parts: Iterable[Mapping[str, Any]]):
        from mooncake.reshard.kv_cache import (
            KVCacheManifest,
            KVCachePlacementManifest,
            KVCachePlacementPart,
        )

        parsed = tuple(KVCachePlacementPart.from_dict(value) for value in parts)
        if not parsed:
            raise KVReshardCompatibilityError("KV reshard placement has no parts")
        first = parsed[0]
        manifest = KVCacheManifest(
            resource_id=first.resource_id,
            revision=first.revision,
            weight_generation=first.weight_generation,
            global_layer_ids=tuple(
                sorted({layer for part in parsed for layer in part.layer_ids})
            ),
            dtype=first.dtype,
            itemsize=first.itemsize,
            page_size=first.page_size,
            total_kv_heads=first.total_kv_heads,
            key_head_dim=first.key_head_dim,
            value_head_dim=first.value_head_dim,
        )
        return KVCachePlacementManifest(manifest=manifest, parts=parsed)

    def complete_local_placement(self):
        if self._complete_placement is not None:
            return self._complete_placement

        from sglang.srt.distributed.utils import get_pp_indices

        from mooncake.reshard.kv_cache import KVCachePlacementPart

        parts = []
        for pp_rank in range(self.pp_size):
            start, end = get_pp_indices(self.total_layers, pp_rank, self.pp_size)
            for tp_rank in range(self.tp_size):
                head_start, head_count, replica_ordinal, replica_count = (
                    _head_placement(self.total_kv_heads, tp_rank, self.tp_size)
                )
                parts.append(
                    KVCachePlacementPart(
                        resource_id=self.resource_id,
                        revision=self.revision,
                        weight_generation=self.weight_generation,
                        participant_id=_participant_id(
                            self.role, self.dp_rank, pp_rank, tp_rank
                        ),
                        instance_id=self.instance_id,
                        dp_rank=self.dp_rank,
                        dp_size=self.dp_size,
                        pp_rank=pp_rank,
                        pp_size=self.pp_size,
                        tp_rank=tp_rank,
                        tp_size=self.tp_size,
                        layer_ids=tuple(range(start, end)),
                        total_kv_heads=self.total_kv_heads,
                        head_start=head_start,
                        head_count=head_count,
                        replica_ordinal=replica_ordinal,
                        replica_count=replica_count,
                        dtype=self.dtype,
                        itemsize=self.itemsize,
                        page_size=self.page_size,
                        key_head_dim=self.key_head_dim,
                        value_head_dim=self.value_head_dim,
                    )
                )
        placement = self.assemble_placement(part.to_dict() for part in parts)
        synthesized_local = placement.part(self.participant_id)
        if synthesized_local.digest != self.local_part.digest:
            raise KVReshardCompatibilityError(
                "SGLang runtime PP layer IDs differ from the synthesized target "
                "placement; check SGLANG_PP_LAYER_PARTITION on every rank"
            )
        self._complete_placement = placement
        return placement

    def plan_decode_routes(
        self,
        *,
        source_placement_json: str,
        routes: Mapping[str, Mapping[str, Any]],
    ) -> KVReshardRoutePlan:
        from mooncake.reshard.kv_cache import (
            KVCachePlacementManifest,
            plan_kv_cache_transfer_to_local_target,
        )

        source = KVCachePlacementManifest.from_json(source_placement_json)
        fanout: Dict[str, int] = {part.participant_id: 0 for part in source.parts}
        local_plan = None
        for target_part in self.complete_local_placement().parts:
            plan = plan_kv_cache_transfer_to_local_target(source, target_part)
            if target_part.participant_id == self.participant_id:
                local_plan = plan
            for writer in plan.source_participant_ids:
                fanout[writer] = fanout.get(writer, 0) + 1
        if local_plan is None:
            raise KVReshardCompatibilityError(
                f"complete Decode placement omits {self.participant_id}"
            )
        infos = []
        for writer in local_plan.source_participant_ids:
            if writer not in routes:
                raise KVReshardCompatibilityError(
                    f"bootstrap route is missing source participant {writer}"
                )
            info = dict(routes[writer])
            info.update(
                {
                    "participant_id": writer,
                    "is_dummy": False,
                    "required_dst_info_num": fanout[writer],
                    "kv_reshard_plan": local_plan.for_source(writer).to_dict(),
                }
            )
            infos.append(info)
        return KVReshardRoutePlan(
            bootstrap_infos=tuple(infos),
            expected_writer_ids=local_plan.expected_writer_ids,
        )

    def prepare_transfer(self, *, logical_plan: Any, target_binding: Any):
        from mooncake.reshard.kv_cache import (
            KVCacheLogicalTransferPlan,
            KVCacheRuntimeBindingManifest,
            prepare_kv_cache_transfer,
        )

        if self.binding is None:
            raise RuntimeError("KV reshard runtime is not physically bound")
        if isinstance(logical_plan, Mapping):
            logical_plan = KVCacheLogicalTransferPlan.from_dict(logical_plan)
        if isinstance(target_binding, Mapping):
            target_binding = KVCacheRuntimeBindingManifest.from_dict(target_binding)
        return prepare_kv_cache_transfer(logical_plan, self.binding, target_binding)

    @staticmethod
    def _physical_slots(
        page_ids: Iterable[int],
        *,
        page_size: int,
        first_page_offset: int,
        token_count: int,
    ) -> np.ndarray:
        pages = np.asarray(tuple(page_ids), dtype=np.int64)
        required_pages = (first_page_offset + token_count + page_size - 1) // page_size
        if pages.ndim != 1 or len(pages) != required_pages:
            raise ValueError("page_ids length does not cover the logical token span")
        if np.any(pages < 0):
            raise ValueError("page_ids must be non-negative")
        offsets = first_page_offset + np.arange(token_count, dtype=np.int64)
        page_positions, in_page = np.divmod(offsets, page_size)
        return pages[page_positions] * page_size + in_page

    def lower_chunk(
        self,
        *,
        prepared_plan: Any,
        source_page_ids: Iterable[int],
        target_page_ids: Iterable[int],
        token_start: int,
        token_count: int,
        max_batch_operations: int = 1024,
    ) -> tuple[KVReshardNativeBatch, ...]:
        from mooncake.reshard.kv_cache import KVCachePreparedTransferPlan

        if self.binding is None or self.transfer_engine is None:
            raise RuntimeError("KV reshard runtime is not physically bound")
        if not isinstance(prepared_plan, KVCachePreparedTransferPlan):
            raise TypeError("prepared_plan must be a KVCachePreparedTransferPlan")
        if max_batch_operations <= 0:
            raise ValueError("max_batch_operations must be positive")
        if token_count <= 0:
            raise ValueError("token_count must be positive")

        first_page_offset = token_start % self.page_size
        source_slots = self._physical_slots(
            source_page_ids,
            page_size=prepared_plan.page_size,
            first_page_offset=first_page_offset,
            token_count=token_count,
        )
        target_slots = self._physical_slots(
            target_page_ids,
            page_size=prepared_plan.page_size,
            first_page_offset=first_page_offset,
            token_count=token_count,
        )
        max_source_slot = int(source_slots.max())
        max_target_slot = int(target_slots.max())

        pending: Dict[
            tuple[str, int | None], tuple[list[int], list[int], list[int]]
        ] = {}
        batches = []

        def flush(key: tuple[str, int | None]) -> None:
            sources, targets, sizes = pending[key]
            if not sizes:
                return
            batches.append(
                KVReshardNativeBatch(
                    endpoint=key[0],
                    source_addresses=tuple(sources),
                    target_addresses=tuple(targets),
                    sizes=tuple(sizes),
                )
            )
            sources.clear()
            targets.clear()
            sizes.clear()

        def append_ops(
            endpoint: str, sources, targets, sizes, batch_limit: int | None
        ) -> None:
            key = (endpoint, batch_limit)
            state = pending.setdefault(key, ([], [], []))
            sources = list(sources)
            targets = list(targets)
            sizes = list(sizes)
            if batch_limit is None:
                state[0].extend(sources)
                state[1].extend(targets)
                state[2].extend(sizes)
                return
            offset = 0
            while offset < len(sizes):
                room = batch_limit - len(state[2])
                take = min(room, len(sizes) - offset)
                state[0].extend(sources[offset : offset + take])
                state[1].extend(targets[offset : offset + take])
                state[2].extend(sizes[offset : offset + take])
                offset += take
                if len(state[2]) == batch_limit:
                    flush(key)

        full_row_starts = full_row_ends = None
        if any(edge.is_full_row for edge in prepared_plan.edges):
            breaks = (
                np.flatnonzero(
                    (np.diff(source_slots) != 1) | (np.diff(target_slots) != 1)
                )
                + 1
            )
            full_row_starts = np.concatenate((np.asarray([0]), breaks))
            full_row_ends = np.concatenate((breaks, np.asarray([token_count])))

        for edge in prepared_plan.edges:
            source_end = (
                max_source_slot * edge.source_row_stride
                + edge.source_head_offset_bytes
                + edge.nbytes
            )
            target_end = (
                max_target_slot * edge.target_row_stride
                + edge.target_head_offset_bytes
                + edge.nbytes
            )
            if source_end > edge.source_capacity:
                raise ValueError("source page map exceeds KV buffer capacity")
            if target_end > edge.target_capacity:
                raise ValueError("target page map exceeds KV buffer capacity")

            if edge.is_full_row:
                assert full_row_starts is not None and full_row_ends is not None
                append_ops(
                    edge.endpoint,
                    (
                        edge.source_base_address
                        + source_slots[full_row_starts] * edge.source_row_stride
                    ).tolist(),
                    (
                        edge.target_base_address
                        + target_slots[full_row_starts] * edge.target_row_stride
                    ).tolist(),
                    ((full_row_ends - full_row_starts) * edge.nbytes).tolist(),
                    None,
                )
            else:
                append_ops(
                    edge.endpoint,
                    (
                        edge.source_base_address
                        + source_slots * edge.source_row_stride
                        + edge.source_head_offset_bytes
                    ).tolist(),
                    (
                        edge.target_base_address
                        + target_slots * edge.target_row_stride
                        + edge.target_head_offset_bytes
                    ).tolist(),
                    [edge.nbytes] * token_count,
                    max_batch_operations,
                )
        for key in tuple(pending):
            flush(key)

        return tuple(batches)

    def submit_chunk(self, batches: Iterable[KVReshardNativeBatch]) -> int:
        if self.transfer_engine is None:
            raise RuntimeError("KV reshard runtime is not physically bound")
        for batch in batches:
            ret = self.transfer_engine.batch_transfer_sync(
                batch.endpoint,
                list(batch.source_addresses),
                list(batch.target_addresses),
                list(batch.sizes),
            )
            if ret != 0:
                return ret
        return 0


def encode_wire_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def decode_wire_json(value: bytes) -> dict[str, Any]:
    payload = json.loads(value.decode())
    if not isinstance(payload, dict):
        raise ValueError("KV_RESHARD_V1 payload must be an object")
    return payload


def record_writer_completion(
    expected_writer_ids: set[str],
    arrived_writer_ids: set[str],
    writer_id: str,
) -> tuple[bool, bool]:
    """Record one idempotent completion and report (accepted, complete)."""
    if writer_id not in expected_writer_ids:
        return False, False
    arrived_writer_ids.add(writer_id)
    return True, arrived_writer_ids == expected_writer_ids


__all__ = [
    "KV_RESHARD_PROTOCOL",
    "KV_RESHARD_SCHEMA_VERSION",
    "KVReshardCompatibilityError",
    "KVReshardRoutePlan",
    "KVReshardRuntime",
    "decode_wire_json",
    "encode_wire_json",
    "record_writer_completion",
]
