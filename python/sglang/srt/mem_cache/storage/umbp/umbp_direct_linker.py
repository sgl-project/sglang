from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.mem_cache.utils import hash_str_to_int64
from sglang.srt.runtime_context import get_memory, get_model
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()

# Keep every control-plane RPC comfortably below gRPC's message-size limit.
# This is a logical-page count; existence queries carry keys only, no ranges.
CHUNK_PAGES = 64

# Budget by ranges because pool layouts attach different counts per object;
# 8192 stays below gRPC's default message limit.
RANGES_PER_CALL = int(os.getenv("UMBP_RANGES_PER_CALL", "8192"))


def _ordered_layers(entry) -> list[int]:
    component_lengths = {len(component) for component in entry.components}
    if len(component_lengths) != 1:
        raise ValueError(
            f"UMBP pool {entry.name} components have different layer counts."
        )
    pool_layer_count = component_lengths.pop()
    if pool_layer_count != len(entry.layer_mapping):
        raise ValueError(
            f"UMBP pool {entry.name} has {pool_layer_count} buffers per component "
            f"but {len(entry.layer_mapping)} mapped layers."
        )
    by_buffer = {
        buffer_index: logical_layer
        for logical_layer, buffer_index in entry.layer_mapping.items()
    }
    if sorted(by_buffer) != list(range(pool_layer_count)):
        raise ValueError(
            f"UMBP pool {entry.name} layer mapping is not a contiguous bijection."
        )
    return [by_buffer[index] for index in range(pool_layer_count)]


def _drain_sync_groups(params: CacheInitParams) -> tuple[Any, ...]:
    """Return the cache-rank groups that must agree on drain state."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return ()

    def wide(group: Any) -> Any | None:
        if group is None or torch.distributed.get_world_size(group=group) <= 1:
            return None
        return group

    attn = tuple(
        group
        for group in (
            wide(params.attn_cp_cache_group),
            wide(params.attn_tp_cache_group),
        )
        if group is not None
    )
    if attn:
        return attn
    tp = wide(params.tp_cache_group)
    return (tp,) if tp is not None else ()


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._producer_index = -1
        self.consumer_index = -1
        self._futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self._producer_index += 1
        self._futures[self._producer_index] = [Future() for _ in range(self.num_layers)]
        return self._producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self._futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self._futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self._futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("UMBP layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self._futures.pop(index, None)

    def reset(self) -> None:
        self._producer_index = -1
        self.consumer_index = -1
        self._futures.clear()


@dataclass
class _PoolRangePlan:
    """Object keys and locations for one pool load."""

    name: PoolName
    keys: list[str]
    locations: list[int]
    entries_per_page: int


def _object_sizes_per_page(entry) -> list[int]:
    """Return per-page object sizes independently of emitted ranges.

    This keeps the tier's exact-tiling validation independent of range generation.
    """
    if entry.packed:
        return [
            sum(size for component in entry.buffer_meta for _, _, size in component)
        ]
    return [sum(size for _, _, size in component) for component in entry.buffer_meta]


def _config_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"UMBP linker config {key!r} must be boolean, got {value!r}.")


def _materialize_cpu_indices(indices: torch.Tensor) -> torch.Tensor:
    """Materialize CPU indices used to derive per-pool row locations."""
    return indices.detach().to(device="cpu", dtype=torch.int64).flatten()


def _parse_storage_extra_config(raw_config):
    # Keep the linker module importable in CPU-only unit tests. The hybrid
    # controller imports device-specific memory-pool modules transitively.
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )

    extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
        raw_config
    )
    return extra_config


class UMBPDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        _storage=None,
    ):
        self.page_size = params.page_size
        self._tp_size = server_args.tp_size
        # Group layers to amortize per-object RPC overhead; 8 is the measured default.
        self.layer_group = max(1, int(os.getenv("UMBP_LAYER_GROUP", "8")))
        if _config_bool(os.getenv("UMBP_LOAD_SPLIT") or "0", "UMBP_LOAD_SPLIT"):
            raise ValueError(
                "UMBP_LOAD_SPLIT is not supported by the dedup-after-insert "
                "load flow: ranks may receive different page sets and deadlock. "
                "The split implementation was removed; see branch "
                "feat/umbp-tree-connector for the previous implementation."
            )

        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self._async_offload_index_snapshot = True
        self._offload_index_fallback_warned = False
        self._offload_index_stream = None
        self._offload_index_done = None
        self._offload_index_device = None
        self._offload_index_buffers: list[torch.Tensor | None] = []
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        tp_rank = 0
        if distributed:
            tp_rank = torch.distributed.get_rank(group=params.tp_cache_group)
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        if self.num_layers <= 0:
            raise ValueError("UMBP requires at least one logical layer.")
        self.pool_layers = {
            name: _ordered_layers(entry) for name, entry in self.pools.items()
        }
        invalid_layers = {
            name: [layer for layer in layers if not 0 <= layer < self.num_layers]
            for name, layers in self.pool_layers.items()
        }
        invalid_layers = {
            name: layers for name, layers in invalid_layers.items() if layers
        }
        if invalid_layers:
            raise ValueError(
                f"UMBP pool mappings contain out-of-range logical layers: {invalid_layers}."
            )
        extra_config = _parse_storage_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        extra_config = dict(extra_config)
        standalone_requested = bool(
            extra_config.get("standalone_address")
            or os.getenv("UMBP_STANDALONE_ADDRESS")
        )
        if "ssd_enabled" in extra_config and _config_bool(
            extra_config["ssd_enabled"], "ssd_enabled"
        ):
            raise ValueError(
                "Direct UMBP requires ssd_enabled=false because its GPU path "
                "cannot use the corresponding host-memory fallback."
            )
        extra_config["ssd_enabled"] = False

        if "cache_remote_fetches" in extra_config and _config_bool(
            extra_config["cache_remote_fetches"], "cache_remote_fetches"
        ):
            raise ValueError(
                "Direct UMBP requires cache_remote_fetches=false because its GPU "
                "path cannot use the corresponding host-memory fallback."
            )
        if standalone_requested:
            extra_config.pop("cache_remote_fetches", None)
        else:
            extra_config["cache_remote_fetches"] = False

        min_object_size = min(
            min(
                pool.get_page_buffer_meta(
                    torch.arange(pool.page_size, dtype=torch.int64)
                )[1]
            )
            for pool in self.pools.values()
        )
        if standalone_requested:
            extra_config.pop("dram_page_size", None)
        else:
            dram_page_size = int(extra_config.get("dram_page_size", min_object_size))
            if not 0 < dram_page_size <= min_object_size:
                raise ValueError(
                    "Direct UMBP requires 0 < dram_page_size <= the smallest "
                    f"per-layer object ({min_object_size} bytes), got {dram_page_size}."
                )
            extra_config["dram_page_size"] = dram_page_size

        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
        )

        if _storage is None:
            from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

            self.storage = UMBPStore(storage_config, mem_pool_host=None)
        else:
            self.storage = _storage

        try:
            client = self.storage.client
            mode = client.get_deployment_mode()
            mode_type = type(mode)
            # Page objects require ranged I/O, currently exposed by StandaloneProcess.
            if mode != mode_type.StandaloneProcess:
                raise ValueError(
                    "Direct UMBP requires the StandaloneProcess deployment mode: "
                    "page-granular objects need ranged multi-buffer I/O, which "
                    f"the {mode!r} client does not implement yet."
                )
            supports_ranged = getattr(client, "supports_ranged_io", None)
            if not callable(supports_ranged) or not bool(supports_ranged()):
                raise ValueError(
                    "Direct UMBP requires a standalone server whose inner backend "
                    "advertises ranged multi-buffer I/O support. Upgrade mori and "
                    "verify the server's distributed/local backend configuration. "
                    "For a Distributed inner backend, set "
                    "UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES to a positive value."
                )
            get_backend_mode = getattr(client, "get_backend_mode", None)
            self.backend_mode = (
                get_backend_mode() if callable(get_backend_mode) else None
            )
            self.deployment_mode = mode
            self._standalone_process_mode = mode == mode_type.StandaloneProcess
            if getattr(self.storage, "_disable_zero_copy_register", False):
                raise ValueError(
                    "Direct UMBP cannot disable zero-copy memory registration."
                )

            self.storage.mem_pool_host = self.pool_group
            self.storage._kv_anchor_is_logical = True
            self.storage.registered_pools = self.pools
            rank_suffix = f"tp{tp_rank}_cp{params.attn_cp_rank}_pp{params.pp_rank}"
            self.storage.mla_suffix = rank_suffix
            self.storage.mha_suffix = rank_suffix
            self._register_buffers()
            logger.info(
                "UMBPDirectLinker topology=standalone_process+%s ranged_io=yes",
                getattr(self.backend_mode, "name", self.backend_mode),
            )
        except BaseException:
            self.storage.close()
            raise

        self._cp_size = params.attn_cp_size
        self._pp_size = params.pp_size
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self._pending: dict[str, list[PoolTransfer]] = {}
        self._gc_frozen = False
        self._load_queue: Queue[tuple[int, list[_PoolRangePlan]] | None] = Queue()
        self._offload_queue: Queue[tuple[list[PoolTransfer], object] | None] = Queue()
        self._offload_results: Queue[bool] = Queue()
        self._offload_sync_groups = _drain_sync_groups(params)
        self._stats = {
            "lookup": 0,
            "load": 0,
            "offload": 0,
            "extkv_reported": 0,
            "extkv_revoked": 0,
            "extkv_reconcile_failures": 0,
        }
        backend_name = getattr(self.backend_mode, "name", str(self.backend_mode))
        self._extkv_enabled = (
            tp_rank == 0
            and backend_name == "Distributed"
            and _config_bool(os.getenv("UMBP_EXTKV_REPORT", "0"), "UMBP_EXTKV_REPORT")
        )
        self._extkv_batch_size = max(
            1, int(os.getenv("UMBP_EXTKV_REPORT_BATCH", "512"))
        )
        self._extkv_flush_seconds = max(
            0.01, float(os.getenv("UMBP_EXTKV_FLUSH_MS", "100")) / 1000.0
        )
        self._extkv_reconcile_seconds = max(
            1.0, float(os.getenv("UMBP_EXTKV_RECONCILE_SECONDS", "60"))
        )
        self._extkv_lock = threading.Lock()
        self._extkv_rpc_lock = threading.Lock()
        self._extkv_pending: set[str] = set()
        self._extkv_reported: set[str] = set()
        self._extkv_required_keys: dict[str, tuple[str, ...]] = {}
        self._extkv_stop = threading.Event()
        self._extkv_thread: threading.Thread | None = None
        self._extkv_tier_dram = None
        if self._extkv_enabled:
            import mori.umbp as _umbp_mod

            self._extkv_tier_dram = _umbp_mod.UMBPTierType.DRAM
        self._load_thread = threading.Thread(
            target=self._load_thread_func,
            daemon=True,
            name=f"umbp-load-tp{tp_rank}",
        )
        self._offload_thread = threading.Thread(
            target=self._offload_thread_func,
            daemon=True,
            name=f"umbp-offload-tp{tp_rank}",
        )
        self._closed = False
        self._load_thread.start()
        self._offload_thread.start()
        if self._extkv_enabled:
            self._extkv_thread = threading.Thread(
                target=self._extkv_thread_func,
                daemon=True,
                name="umbp-extkv-reconcile",
            )
            self._extkv_thread.start()

    def _register_buffers(self) -> None:
        seen = set()
        self._registered: list[tuple[int, int]] = []
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                if not self.storage.client.register_memory(*allocation):
                    raise RuntimeError(
                        "Failed to register a GPU KV buffer with UMBP: "
                        f"ptr=0x{allocation[0]:x}, size={allocation[1]}."
                    )
                self._registered.append(allocation)

    def _object_keys_for_pages(
        self, page_keys: list[str], transfer: PoolTransfer
    ) -> tuple[list[str], int]:
        component_keys, multiplier = self.storage._get_hybrid_page_component_keys(
            page_keys, transfer
        )
        entry = self.pools[transfer.name]
        # One key names one stored object, and a packed entry stores a whole
        # page as one object regardless of how many components it has.
        entries_per_page = 1 if entry.packed else len(entry.components)
        if multiplier != entries_per_page:
            raise ValueError(
                f"UMBP pool {transfer.name} produced {multiplier} keys per page "
                f"but its layout yields {entries_per_page} objects per page "
                f"(packed={entry.packed}, components={len(entry.components)})."
            )
        # No layer suffix: one object per page (or per page component) holds
        # every layer, and a layer is read back as a byte range inside it.
        return component_keys, multiplier

    def _object_keys(self, transfer: PoolTransfer) -> list[str]:
        keys, _ = self._object_keys_for_pages(list(transfer.keys or []), transfer)
        return keys

    def _page_exists(self, page_keys: list[str], transfer: PoolTransfer) -> list[bool]:
        entry = self.pools[transfer.name]
        objects_per_page = 1 if entry.packed else len(entry.components)
        max_objects = CHUNK_PAGES * self.num_layers
        pages_per_call = max(1, max_objects // objects_per_page)

        page_exists = []
        for start in range(0, len(page_keys), pages_per_call):
            chunk_pages = page_keys[start : start + pages_per_call]
            object_keys, _ = self._object_keys_for_pages(chunk_pages, transfer)
            exists = list(self.storage.client.batch_exists(object_keys))
            if len(exists) != len(object_keys):
                raise RuntimeError(
                    f"UMBP exists result-size mismatch for pool {transfer.name}: "
                    f"expected={len(object_keys)} actual={len(exists)}."
                )
            page_exists.extend(
                all(exists[index : index + objects_per_page])
                for index in range(0, len(exists), objects_per_page)
            )
        return page_exists

    @staticmethod
    def _apply_hit_policy(
        valid_pages: list[int], page_exists: list[bool], transfer: PoolTransfer
    ) -> list[int]:
        present_prefix = [0]
        for present in page_exists:
            present_prefix.append(present_prefix[-1] + int(present))

        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            return [end for end in valid_pages if present_prefix[end] == end]
        if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys or ()))
            return [
                end
                for end in valid_pages
                if present_prefix[end] - present_prefix[max(0, end - trailing)]
                == end - max(0, end - trailing)
            ]
        raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys or [])
        if not page_keys:
            return []

        valid_pages = list(range(1, len(page_keys) + 1))
        for transfer in expanded:
            page_exists = self._page_exists(page_keys, transfer)
            valid_pages = self._apply_hit_policy(valid_pages, page_exists, transfer)
            if not valid_pages:
                break

        self._stats["lookup"] += 1
        if valid_pages:
            logger.debug(
                "UMBP direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                valid_pages[-1],
                len(valid_pages),
            )
        return valid_pages

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Lookup establishes a restorable boundary before insert de-duplicates
        # resident pages. The remaining transfer can therefore contain only a
        # side pool such as SWA, with no KV transfer at all.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self._pending:
            raise RuntimeError(f"UMBP load for rid={rid} is already queued.")
        self._pending[rid] = expanded
        return True

    def start_layer_wise_loading(self) -> int:
        if not self._pending:
            return -1
        self._freeze_gc_once()
        pending = self._pending
        self._pending = {}
        plans = self._build_load_plans(list(pending.values()))
        counter_index = self.layer_done_counter.update_producer()
        self._load_queue.put((counter_index, plans))
        self._stats["load"] += len(pending)
        return counter_index

    def _build_load_plans(
        self,
        request_transfers: list[list[PoolTransfer]],
        *,
        materialize_indices: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> list[_PoolRangePlan]:
        """Build a batch plan shared by load and offload."""
        grouped: dict[PoolName, list[PoolTransfer]] = {}
        for transfers in request_transfers:
            for transfer in transfers:
                grouped.setdefault(transfer.name, []).append(transfer)

        plans = []
        # One logical source can fan out to several physical pools (for
        # example GLM's KV and INDEXER pools). Snapshot it once, then let each
        # pool independently validate and derive its own row geometry.
        cpu_indices: dict[int, torch.Tensor] = {}
        for name, transfers in grouped.items():
            entry = self.pools[name]
            entries_per_page = 1 if entry.packed else len(entry.components)
            keys: list[str] = []
            locations: list[int] = []
            for transfer in transfers:
                page_keys = list(transfer.keys or [])
                transfer_keys, multiplier = self._object_keys_for_pages(
                    page_keys, transfer
                )
                if multiplier != entries_per_page:
                    raise ValueError(
                        f"UMBP pool {name} emits {multiplier} keys per page but "
                        f"its layout yields {entries_per_page} objects per page "
                        f"(packed={entry.packed})."
                    )
                if len(transfer_keys) != len(page_keys) * entries_per_page:
                    raise ValueError(
                        f"UMBP pool {name} key count mismatch: "
                        f"keys={len(transfer_keys)} pages={len(page_keys)}."
                    )
                keys.extend(transfer_keys)
                indices = transfer.host_indices
                if indices is None:
                    raise ValueError(f"UMBP pool {name} transfer has no indices.")
                source_id = id(indices)
                prepared_indices = cpu_indices.get(source_id)
                if prepared_indices is None:
                    prepared_indices = (
                        materialize_indices(indices)
                        if materialize_indices is not None
                        else _materialize_cpu_indices(indices)
                    )
                    cpu_indices[source_id] = prepared_indices
                locations.extend(entry.prepare_locations(prepared_indices))
            if len(keys) != len(locations) * entries_per_page:
                raise ValueError(
                    f"UMBP pool {name} plan mismatch: keys={len(keys)} "
                    f"rows={len(locations)} per_page={entries_per_page}."
                )
            plans.append(_PoolRangePlan(name, keys, locations, entries_per_page))

        if not plans or not plans[0].keys:
            raise ValueError("Layer-wise UMBP load has no object keys.")
        return plans

    def _materialize_offload_indices(
        self, indices: torch.Tensor, slot: int
    ) -> torch.Tensor:
        if not self._async_offload_index_snapshot or indices.device.type == "cpu":
            return _materialize_cpu_indices(indices)
        if (
            indices.dtype != torch.int64
            or indices.ndim != 1
            or not indices.is_contiguous()
        ):
            return self._fallback_offload_indices(indices, "tensor shape or dtype")
        if (
            self._offload_index_device is not None
            and indices.device != self._offload_index_device
        ):
            return self._fallback_offload_indices(indices, "device mismatch")

        try:
            with device_module.device(indices.device):
                if self._offload_index_stream is None:
                    self._offload_index_device = indices.device
                    self._offload_index_stream = device_module.Stream(
                        device=indices.device
                    )
                    self._offload_index_done = device_module.Event()
                while len(self._offload_index_buffers) <= slot:
                    self._offload_index_buffers.append(None)
                count = indices.numel()
                buffer = self._offload_index_buffers[slot]
                if buffer is None or buffer.numel() < count:
                    buffer = self._allocate_offload_index_buffer(count)
                    self._offload_index_buffers[slot] = buffer
                source = indices.detach()
                with device_module.stream(self._offload_index_stream):
                    # Register before enqueue so a failed copy cannot leave an
                    # untracked side-stream read of the source allocation.
                    source.record_stream(self._offload_index_stream)
                    buffer[:count].copy_(source, non_blocking=True)
                    self._offload_index_done.record(self._offload_index_stream)
                self._offload_index_done.synchronize()
                return buffer[:count]
        except RuntimeError:
            self._async_offload_index_snapshot = False
            logger.exception(
                "UMBP async index snapshot failed; falling back to synchronous D2H"
            )
            return _materialize_cpu_indices(indices)

    @staticmethod
    def _allocate_offload_index_buffer(count: int) -> torch.Tensor:
        return torch.empty(count, dtype=torch.int64, device="cpu", pin_memory=True)

    def _fallback_offload_indices(
        self, indices: torch.Tensor, reason: str
    ) -> torch.Tensor:
        if not self._offload_index_fallback_warned:
            self._offload_index_fallback_warned = True
            logger.warning(
                "UMBP offload index snapshot is using synchronous fallback: "
                "%s (device=%s dtype=%s shape=%s contiguous=%s)",
                reason,
                indices.device,
                indices.dtype,
                tuple(indices.shape),
                indices.is_contiguous(),
            )
        return _materialize_cpu_indices(indices)

    def _layer_ranges(self, plan: _PoolRangePlan, logical_layer: int):
        """(ptrs, sizes, offsets) for one layer, one nested list per object."""
        meta = self.pools[plan.name].get_prepared_layer_range_meta(
            plan.locations, logical_layer
        )
        if meta is None:
            return None
        ptrs, sizes, offsets = meta
        if not len(ptrs) == len(sizes) == len(offsets) == len(plan.keys):
            raise ValueError(
                f"UMBP pool {plan.name} layer {logical_layer} produced "
                f"{len(ptrs)} range entries for {len(plan.keys)} keys."
            )
        return ptrs, sizes, offsets

    def _load_thread_func(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, plans = task
                self._run_layer_wise_batch(counter_index, plans)
            finally:
                self._load_queue.task_done()

    def _all_layer_ranges(self, plan: _PoolRangePlan):
        """Every layer's ranges, accumulated per object.

        Offload requires one call to carry ranges that tile the object exactly,
        so an object's ranges must never be split across calls.
        """
        ptrs: list[list[int]] = [[] for _ in plan.keys]
        sizes: list[list[int]] = [[] for _ in plan.keys]
        offsets: list[list[int]] = [[] for _ in plan.keys]
        for logical_layer in self.pool_layers[plan.name]:
            meta = self._layer_ranges(plan, logical_layer)
            if meta is None:
                continue
            layer_ptrs, layer_sizes, layer_offsets = meta
            for index in range(len(plan.keys)):
                ptrs[index].extend(layer_ptrs[index])
                sizes[index].extend(layer_sizes[index])
                offsets[index].extend(layer_offsets[index])
        return ptrs, sizes, offsets

    @staticmethod
    def _plans_covering(
        by_layer: dict[int, list[_PoolRangePlan]], group: list[int]
    ) -> list[_PoolRangePlan]:
        """Plans touching any layer of the group, each listed once, in order."""
        seen: set[int] = set()
        plans = []
        for logical_layer in group:
            for plan in by_layer.get(logical_layer, ()):
                if id(plan) in seen:
                    continue
                seen.add(id(plan))
                plans.append(plan)
        return plans

    def _layer_group_ranges(self, plan: _PoolRangePlan, layers: list[int]):
        """One group of layers' ranges, accumulated per object.

        `_all_layer_ranges` for a slice of the layer stack. Returns None when
        this pool covers none of the group, so the caller can skip the call.
        """
        ptrs: list[list[int]] = [[] for _ in plan.keys]
        sizes: list[list[int]] = [[] for _ in plan.keys]
        offsets: list[list[int]] = [[] for _ in plan.keys]
        covered = 0
        for logical_layer in layers:
            meta = self._layer_ranges(plan, logical_layer)
            if meta is None:
                continue
            covered += 1
            layer_ptrs, layer_sizes, layer_offsets = meta
            for index in range(len(plan.keys)):
                ptrs[index].extend(layer_ptrs[index])
                sizes[index].extend(layer_sizes[index])
                offsets[index].extend(layer_offsets[index])
        if covered == 0:
            return None
        return ptrs, sizes, offsets

    @staticmethod
    def _entries_per_call(sizes: list[list[int]]) -> int:
        """Objects per RPC, budgeted by the ranges they actually carry.

        Counted from the ranges that were built, not from the layer count. A
        packed pool puts one range per component per layer on an object, so a
        packed K/V pool carries twice what the layer count suggests and the
        budget would be overshot by that factor.
        """
        ranges_per_object = max((len(entry) for entry in sizes), default=1)
        return max(1, RANGES_PER_CALL // max(1, ranges_per_object))

    def _run_layer_wise_batch(
        self, counter_index: int, plans: list[_PoolRangePlan]
    ) -> None:
        try:
            by_layer: dict[int, list[_PoolRangePlan]] = defaultdict(list)
            for plan in plans:
                for logical_layer in self.pool_layers[plan.name]:
                    by_layer[logical_layer].append(plan)

            for group in self._layer_groups():
                for plan in self._plans_covering(by_layer, group):
                    meta = self._layer_group_ranges(plan, group)
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    step = self._entries_per_call(sizes)
                    for start in range(0, len(plan.keys), step):
                        end = start + step
                        chunk_keys = plan.keys[start:end]
                        results = list(
                            self.storage.client.batch_get_ranges_into_ptr(
                                chunk_keys,
                                ptrs[start:end],
                                sizes[start:end],
                                offsets[start:end],
                            )
                        )
                        if len(results) != len(chunk_keys) or not all(results):
                            where = (
                                f"layer={group[0]}"
                                if len(group) == 1
                                else f"layers={group[0]}..{group[-1]}"
                            )
                            raise RuntimeError(
                                f"UMBP get failed for pool={plan.name}, {where}: "
                                f"success={sum(bool(value) for value in results)}/"
                                f"{len(chunk_keys)}."
                            )
                # Only now is every layer in the group readable, so they are
                # released together. A group wider than 1 trades overlap
                # granularity for fewer times each object is named on the wire.
                for logical_layer in group:
                    self.layer_done_counter.complete(counter_index, logical_layer)
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("UMBP layer-wise load batch failed")

    def _layer_groups(self) -> list[list[int]]:
        return [
            list(range(start, min(start + self.layer_group, self.num_layers)))
            for start in range(0, self.num_layers, self.layer_group)
        ]

    def _extkv_page_hashes(self, expanded: list[PoolTransfer]) -> list[str]:
        """Return the logical KV page hashes exactly once for an offload task."""
        if not self._extkv_enabled:
            return []
        for transfer in expanded:
            entry = self.pools[transfer.name]
            if entry.indices_from_pool == PoolName.KV and transfer.keys:
                return list(transfer.keys)
        return []

    def _extkv_keys_for_page(self, page_hash: str) -> tuple[str, ...]:
        """Build every object key required to restore one logical block."""
        required: list[str] = []
        for pp_rank in range(self._pp_size):
            for cp_rank in range(self._cp_size):
                for tp_rank in range(self._tp_size):
                    rank_suffix = f"tp{tp_rank}_cp{cp_rank}_pp{pp_rank}"
                    for pool_name in self.pools:
                        keys, _ = self.storage._get_hybrid_page_component_keys(
                            [page_hash],
                            PoolTransfer(name=pool_name),
                            rank_suffix=rank_suffix,
                        )
                        required.extend(keys)
        # Stable de-duplication protects against two logical entries resolving
        # to the same physical object without changing the AND predicate.
        return tuple(dict.fromkeys(required))

    def _queue_extkv_pages(self, page_hashes: list[str]) -> None:
        if not self._extkv_enabled or not page_hashes:
            return
        # Serialize state publication against reconcile/revoke. Otherwise a new
        # offload can observe "reported", skip enqueueing, then lose the race to
        # a concurrent reconcile that revokes and removes the same hash.
        with self._extkv_rpc_lock:
            with self._extkv_lock:
                for page_hash in page_hashes:
                    external_hash = str(hash_str_to_int64(page_hash))
                    if external_hash not in self._extkv_required_keys:
                        self._extkv_required_keys[external_hash] = (
                            self._extkv_keys_for_page(page_hash)
                        )
                    if external_hash not in self._extkv_reported:
                        self._extkv_pending.add(external_hash)
                should_flush = len(self._extkv_pending) >= self._extkv_batch_size
        if should_flush:
            self._flush_extkv()

    def _flush_extkv(self) -> None:
        if not self._extkv_enabled:
            return
        with self._extkv_rpc_lock:
            with self._extkv_lock:
                hashes = list(self._extkv_pending)[: self._extkv_batch_size]
            if not hashes:
                return
            try:
                ok = self.storage.client.report_external_kv_blocks(
                    hashes, self._extkv_tier_dram
                )
            except BaseException:
                logger.exception("UMBP external-KV report RPC failed")
                return
            if not ok:
                logger.warning(
                    "UMBP external-KV report failed for %d hashes", len(hashes)
                )
                return
            with self._extkv_lock:
                self._extkv_pending.difference_update(hashes)
                self._extkv_reported.update(hashes)
            self._stats["extkv_reported"] += len(hashes)

    def _reconcile_extkv_chunk(self, hashes: list[str]) -> None:
        # Keep the exists -> revoke -> state-publication transaction serialized
        # with report/queue, but only for one bounded RPC chunk. Holding this
        # lock for the entire reconcile pass can stall the offload worker behind
        # dozens of control-plane round trips.
        with self._extkv_rpc_lock:
            flattened: list[str] = []
            spans: list[tuple[str, int, int]] = []
            with self._extkv_lock:
                for external_hash in hashes:
                    keys = self._extkv_required_keys.get(external_hash, ())
                    start = len(flattened)
                    flattened.extend(keys)
                    spans.append((external_hash, start, len(flattened)))
            if not flattened:
                return
            try:
                exists = list(self.storage.client.batch_exists(flattened))
            except BaseException:
                self._stats["extkv_reconcile_failures"] += 1
                logger.exception("UMBP external-KV exists RPC failed")
                return
            if len(exists) != len(flattened):
                self._stats["extkv_reconcile_failures"] += 1
                logger.warning(
                    "UMBP external-KV reconcile result mismatch: expected=%d actual=%d",
                    len(flattened),
                    len(exists),
                )
                return
            dead = [
                external_hash
                for external_hash, start, end in spans
                if start == end or not all(exists[start:end])
            ]
            if not dead:
                return
            try:
                ok = self.storage.client.revoke_external_kv_blocks(
                    dead, self._extkv_tier_dram
                )
            except BaseException:
                self._stats["extkv_reconcile_failures"] += 1
                logger.exception("UMBP external-KV revoke RPC failed")
                return
            if not ok:
                self._stats["extkv_reconcile_failures"] += 1
                logger.warning(
                    "UMBP external-KV revoke failed for %d hashes", len(dead)
                )
                return
            with self._extkv_lock:
                self._extkv_reported.difference_update(dead)
                for external_hash in dead:
                    if external_hash not in self._extkv_pending:
                        self._extkv_required_keys.pop(external_hash, None)
            self._stats["extkv_revoked"] += len(dead)

    def _reconcile_extkv(self) -> None:
        if not self._extkv_enabled:
            return
        with self._extkv_lock:
            reported = list(self._extkv_reported)
            required = dict(self._extkv_required_keys)
        chunk: list[str] = []
        key_count = 0
        for external_hash in reported:
            count = len(required.get(external_hash, ()))
            if chunk and key_count + count > RANGES_PER_CALL:
                self._reconcile_extkv_chunk(chunk)
                chunk = []
                key_count = 0
            chunk.append(external_hash)
            key_count += count
        if chunk:
            self._reconcile_extkv_chunk(chunk)

    def _clear_extkv(self) -> None:
        if not self._extkv_enabled:
            return
        with self._extkv_rpc_lock:
            try:
                ok = self.storage.client.revoke_all_external_kv_blocks_at_tier(
                    self._extkv_tier_dram
                )
            except BaseException:
                ok = False
                logger.exception("UMBP external-KV revoke-all RPC failed")
            if not ok:
                logger.warning("UMBP external-KV revoke-all failed during reset/close")
            with self._extkv_lock:
                self._extkv_pending.clear()
                self._extkv_reported.clear()
                self._extkv_required_keys.clear()

    def _extkv_thread_func(self) -> None:
        next_reconcile = time.monotonic() + self._extkv_reconcile_seconds
        while not self._extkv_stop.wait(self._extkv_flush_seconds):
            try:
                self._flush_extkv()
                now = time.monotonic()
                if now >= next_reconcile:
                    self._reconcile_extkv()
                    next_reconcile = now + self._extkv_reconcile_seconds
            except BaseException:
                self._stats["extkv_reconcile_failures"] += 1
                logger.exception("UMBP external-KV background task failed")

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self._freeze_gc_once()
        ready_event = device_module.Event()
        ready_event.record()
        self._offload_queue.put((expanded, ready_event))
        return True

    def _offload_thread_func(self) -> None:
        while True:
            task = self._offload_queue.get()
            try:
                if task is None:
                    return
                expanded, ready_event = task
                page_hashes = self._extkv_page_hashes(expanded)
                try:
                    success = self._run_offload(expanded, ready_event)
                except BaseException:
                    logger.exception("UMBP offload failed")
                    success = False
                if success:
                    self._queue_extkv_pages(page_hashes)
                self._offload_results.put(success)
            finally:
                self._offload_queue.task_done()

    def _run_offload(self, expanded: list[PoolTransfer], ready_event: object) -> bool:
        ready_event.synchronize()
        next_slot = 0

        def materialize_indices(indices: torch.Tensor) -> torch.Tensor:
            nonlocal next_slot
            slot = next_slot
            next_slot += 1
            return self._materialize_offload_indices(indices, slot)

        plans = self._build_load_plans(
            [expanded], materialize_indices=materialize_indices
        )
        plans_by_name = {plan.name: plan for plan in plans}
        for transfer in expanded:
            plan = plans_by_name[transfer.name]
            entry = self.pools[transfer.name]
            # From the pool layout, never from the ranges below: see
            # _object_sizes_per_page.
            per_page = _object_sizes_per_page(entry)
            if len(per_page) != plan.entries_per_page:
                raise ValueError(
                    f"UMBP pool {transfer.name} declares {len(per_page)} object "
                    f"sizes per page but yields {plan.entries_per_page} objects."
                )
            object_sizes = [
                per_page[index % plan.entries_per_page]
                for index in range(len(plan.keys))
            ]
            ptrs, sizes, offsets = self._all_layer_ranges(plan)

            # Preserve page order so the tier can collapse a layer into a strided copy.

            # An object's ranges must tile it exactly, so a chunk boundary may
            # fall between objects but never inside one.
            step = self._entries_per_call(sizes)
            for start in range(0, len(plan.keys), step):
                end = start + step
                chunk_keys = plan.keys[start:end]
                results = list(
                    self.storage.client.batch_put_ranges_from_ptr(
                        chunk_keys,
                        object_sizes[start:end],
                        ptrs[start:end],
                        sizes[start:end],
                        offsets[start:end],
                    )
                )
                if len(results) != len(chunk_keys) or not all(results):
                    logger.warning(
                        "UMBP offload failed: pool=%s object_range=[%d,%d) "
                        "success=%d/%d returned=%d",
                        transfer.name,
                        start,
                        min(end, len(plan.keys)),
                        sum(bool(value) for value in results),
                        len(chunk_keys),
                        len(results),
                    )
                    return False

        self._stats["offload"] += 1
        return True

    def _freeze_gc_once(self) -> None:
        if self._gc_frozen:
            return
        freeze_gc("UMBP direct linker")
        self._gc_frozen = True

    def num_completed_offloads(self) -> int:
        # The tree agrees on the drain count across ranks before calling pop.
        return self._offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        local_result = self._offload_results.get_nowait()
        return bool(self._reduce_min_offload_state([local_result])[0])

    def _reduce_min_offload_state(self, values: list) -> list:
        if not self._offload_sync_groups:
            return [int(value) for value in values]
        synced = torch.tensor([int(value) for value in values], dtype=torch.int)
        for group in self._offload_sync_groups:
            torch.distributed.all_reduce(
                synced, op=torch.distributed.ReduceOp.MIN, group=group
            )
        return synced.tolist()

    def reset(self) -> None:
        self._pending.clear()
        self._load_queue.join()
        self._offload_queue.join()
        self._clear_extkv()
        while True:
            try:
                self._offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        for thread, queue in (
            (self._offload_thread, self._offload_queue),
            (self._load_thread, self._load_queue),
        ):
            if thread.is_alive():
                queue.put(None)
                thread.join()
        if self._extkv_thread is not None:
            self._extkv_stop.set()
            self._extkv_thread.join()
            self._extkv_thread = None
        if self._standalone_process_mode and self._registered:
            # StandaloneProcess deregistration is client-wide; one call tears
            # down every registered region. Keep the GPU tensors alive until
            # the synchronous RPC has completed successfully.
            self.storage.client.deregister_memory(self._registered[0][0])
        logger.info("UMBP direct linker stats: %s", self._stats)
        self.storage.close()
        self._closed = True
