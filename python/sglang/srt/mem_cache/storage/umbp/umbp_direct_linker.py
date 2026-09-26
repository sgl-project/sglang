from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
from hashlib import blake2b
from queue import Empty, Queue
from typing import Any, Callable

import numpy as np
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
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()

# Keep every control-plane RPC comfortably below gRPC's message-size limit.
# This is a logical-page count; existence queries carry keys only, no ranges.
CHUNK_PAGES = 64

# Budget by ranges because pool layouts attach different counts per object;
# 8192 stays below gRPC's default message limit.
RANGES_PER_CALL = int(os.getenv("UMBP_RANGES_PER_CALL", "8192"))

# Bound retained lookup metadata and the CPU page-agreement buffer.
SPLIT_LOOKUP_CACHE = 128
SPLIT_MAX_PAGES = 65536


def _split_windows(pages: int, world: int) -> tuple[tuple[int, int], ...]:
    width = -(-pages // world)
    return tuple(
        (min(rank * width, pages), min((rank + 1) * width, pages))
        for rank in range(world)
    )


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    # A rank-replicated group (MLA / DSA) holds byte-identical pages on every
    # attention TP rank, so a tp term stores tp_size copies of the same page.
    # Only the key collapses -- every rank still writes, because a Local-mode
    # tier is private to its rank and a standalone server is per node.
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


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


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int, on_group_ready=None):
        self.num_layers = num_layers
        self._on_group_ready = on_group_ready
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
            if self._on_group_ready is not None:
                self._on_group_ready(index, threshold)
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
    # Only common pages, in the agreed order; I/O also includes local remainders.
    all_locations: list[int] | None = None
    windows: tuple[tuple[int, int], ...] = ()


@dataclass
class _SplitShare:
    common: list[list[str]]
    owned: list[set[str]]
    windows: tuple[tuple[int, int], ...]


@dataclass
class _SplitLoad:
    plans: dict[PoolName, _PoolRangePlan]
    exchanged: int = -1
    failure: BaseException | None = None
    # Batch-level agreement, accumulated on device and read once, at the last
    # layer group. Reading it per group blocks the forward thread behind every
    # kernel already queued on the stream, which costs far more than the
    # exchange itself.
    status: torch.Tensor | None = None
    rows: dict[tuple[PoolName, int], torch.Tensor] = field(default_factory=dict)


# One queued offload: the pools it resolved to, and the event guarding its KV.
_OffloadTask = tuple[list[PoolTransfer], object]


def _offload_task_pages(expanded: list[PoolTransfer]) -> int:
    """Pages this task puts into its widest pool, which is what sizes a plan."""
    return max((len(transfer.keys or ()) for transfer in expanded), default=0)


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
        # Group layers to amortize per-object RPC overhead; 8 is the measured default.
        self.layer_group = max(1, int(os.getenv("UMBP_LAYER_GROUP", "8")))
        # Coalesce queued offload tasks up to this many pages; offload_nodes
        # queues one task per node, so they arrive a page or two at a time.
        self._offload_coalesce_pages = max(
            1, int(os.getenv("UMBP_OFFLOAD_COALESCE_PAGES", "1024"))
        )
        split_requested = _config_bool(
            os.getenv("UMBP_LOAD_SPLIT") or "0", "UMBP_LOAD_SPLIT"
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
        rank_replicated = self.pool_group.rank_replicated
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
        self._split_load = False
        self._split_state: dict[int, _SplitLoad] = {}
        self._lookup_pages: dict[str, list[str]] = {}
        self._pending_pages: dict[str, list[str] | None] = {}
        if split_requested:
            self._init_split(params, tp_rank)
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
            tp_size=get_parallel().tp_size,
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

            # per_rank_keyspace: every object key this class writes carries a
            # tp{rank} suffix (set just below), so the store must not put the
            # ranks into the shared-SSD leader/follower scheme meant for
            # deduplicating replicated MLA KV.
            self.storage = UMBPStore(
                storage_config, mem_pool_host=None, per_rank_keyspace=True
            )
        else:
            self.storage = _storage

        try:
            client = self.storage.client
            mode = client.get_deployment_mode()
            mode_type = type(mode)
            # What page-granular objects actually need is ranged multi-buffer
            # I/O. That used to be true only of StandaloneProcess, so the gate
            # was written as a mode test -- but mori now implements it in the
            # in-process client too ("Both media behind LocalStorageManager
            # implement ranged I/O now", standalone_client.h), and a mode test
            # would keep rejecting a client that can do the job.
            #
            # So ask the client what it supports instead of inferring it from
            # which mode it is. supports_ranged_io() is the capability this
            # code depends on, it is already consulted below, and a client that
            # answers truthfully cannot be wrongly admitted by it.
            supports_ranged = getattr(client, "supports_ranged_io", None)
            if not callable(supports_ranged) or not bool(supports_ranged()):
                raise ValueError(
                    f"Direct UMBP needs ranged multi-buffer I/O, which this "
                    f"{mode!r} client does not advertise. Upgrade mori; if the "
                    "server has a Distributed inner backend, set "
                    "UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES to a positive value "
                    "(both scratch arenas must be non-zero), and note that a "
                    "read-only SharedSSDFollower is whole-object by design."
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
            rank_suffix = _storage_suffix(
                rank_replicated=rank_replicated,
                tp_rank=tp_rank,
                attn_cp_rank=params.attn_cp_rank,
                pp_rank=params.pp_rank,
            )
            self.storage.mla_suffix = rank_suffix
            self.storage.mha_suffix = rank_suffix
            self._register_buffers()
            # Report the mode rather than asserting it in the text: the line is
            # what every acceptance check greps to prove the linker attached at
            # all, and it used to say "standalone_process" unconditionally, so
            # it could not have shown an embedded run for what it was.
            logger.info(
                "UMBPDirectLinker topology=%s+%s ranged_io=yes "
                "rank_replicated=%s suffix=%s",
                mode.name,
                self.backend_mode.name if self.backend_mode is not None else None,
                rank_replicated,
                rank_suffix,
            )
        except BaseException:
            self.storage.close()
            raise

        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            self._exchange_ready_groups if self._split_load else None,
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        # 0 disables the gate. A positive value vetoes split batches while the
        # tier is too idle for the split to pay for itself; the opening minutes
        # are ignored because warmup is not the steady state it calibrates to.
        self._split_min_rate = float(os.getenv("UMBP_LOAD_SPLIT_MIN_RESTORE_RATE", "0"))
        self._split_rate_min_seconds = float(
            os.getenv("UMBP_LOAD_SPLIT_RATE_MIN_SECONDS", "300")
        )
        self._split_rate_start = time.monotonic()
        self._split_loads_seen = 0
        self._pending: dict[str, list[PoolTransfer]] = {}
        self._gc_frozen = False
        self._load_queue: Queue[
            tuple[int, list[str], list[_PoolRangePlan], object] | None
        ] = Queue()
        self._completed_loads: Queue[list[str]] = Queue()
        self._offload_queue: Queue[tuple[list[PoolTransfer], object] | None] = Queue()
        self._offload_results: Queue[bool] = Queue()
        self._stats = {
            "lookup": 0,
            "load": 0,
            "offload": 0,
            # offload / offload_batches is the coalescing actually achieved.
            "offload_batches": 0,
        }
        if self._split_load:
            self._stats.update(
                split_batches=0,
                # Rank-local: only nonempty candidate batches count as skipped.
                split_skipped_batches=0,
                split_divergent_pages=0,
                split_local_pages=0,
                # Kept apart from split_skipped_batches so a deliberate veto
                # never reads as a failure.
                split_rate_gated=0,
                split_rate_milli=0,
            )
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

    def _init_split(self, params: CacheInitParams, tp_rank: int) -> None:
        parallel = get_parallel()
        # rank_replicated is the same predicate the platform's other MLA paths
        # use to share bytes across ranks. CP/DP/PP and disagg decode are
        # rejected for rank numbering and lockstep, not for replication.
        if (
            not self.pool_group.rank_replicated
            or parallel.attn_cp_size != 1
            or parallel.enable_dp_attention
            or parallel.pp_size != 1
            or get_disagg().disaggregation_mode == "decode"
        ):
            raise ValueError(
                "UMBP_LOAD_SPLIT requires rank-replicated KV, CP=PP=1, "
                "no DP attention, and no disaggregation decode."
            )
        # KV-source pools only: resolve_transfers forces ALL_PAGES on exactly
        # these, so every page in the agreed prefix exists. A SWA-source pool
        # holds a trailing window, and reading before it fails the restore.
        self._split_pools = [
            name
            for name, entry in self.pools.items()
            if entry.indices_from_pool == PoolName.KV
        ]
        from sglang.srt.distributed.parallel_state import get_attn_tp_group

        self._split_pg = get_attn_tp_group().device_group
        self._split_rank = torch.distributed.get_rank(self._split_pg)
        self._split_world = torch.distributed.get_world_size(self._split_pg)
        if (self._split_rank, self._split_world) != (tp_rank, parallel.tp_size):
            raise ValueError("UMBP split exchange group must match the TP keyspace.")
        if self._split_world == 1:
            return
        self._split_cpu_group = (
            params.attn_tp_cache_group
            if params.attn_tp_cache_group is not None
            else params.tp_cache_group
        )
        self._split_min_pages = max(
            1, int(os.getenv("UMBP_LOAD_SPLIT_MIN_PAGES", "16"))
        )
        self._split_max_rids = max(1, int(os.getenv("UMBP_LOAD_SPLIT_MAX_RIDS", "8")))
        capacity = int(os.getenv("UMBP_LOAD_SPLIT_STAGING_MIB", "512")) << 20
        if capacity <= 0:
            raise ValueError("UMBP_LOAD_SPLIT_STAGING_MIB must be positive.")
        for name in self._split_pools:
            for component in self.pools[name].buffer_meta:
                for _, stride, size in component:
                    if stride <= 0 or size % stride:
                        raise ValueError(
                            f"UMBP split pool {name} has invalid row geometry."
                        )
        device = self.pools[self._split_pools[0]].components[0][0].device
        # Only forward touches this buffer, on its compute stream. It is never
        # an I/O destination, so it needs neither registration nor loader fences.
        self._split_recv = torch.empty(capacity, dtype=torch.uint8, device=device)
        self._split_status = torch.empty(1, dtype=torch.int64, device=device)
        self._split_agreement = torch.empty(
            1 + 4 * self._split_max_rids, dtype=torch.int64
        )
        self._split_load = True
        logger.info(
            "UMBP split load requested: rank=%d/%d min_pages=%d staging=%.1f MiB",
            self._split_rank,
            self._split_world,
            self._split_min_pages,
            capacity / (1 << 20),
        )

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
        if self._split_load:
            self._lookup_pages.pop(rid, None)
            # Retain the existing snapshot. A match need not queue a load;
            # build their inverse map only after admission, before the vote.
            self._lookup_pages[rid] = page_keys
            if len(self._lookup_pages) > SPLIT_LOOKUP_CACHE:
                self._lookup_pages.pop(next(iter(self._lookup_pages)))

        valid_pages = list(range(1, len(page_keys) + 1))
        for transfer in expanded:
            # Probe only as far as the surviving boundary: no hit policy reads
            # past its own end offset, so pages beyond the longest candidate
            # cannot change the answer. This runs synchronously inside the
            # scheduler's prefill batch build, and DP-attention ranks are
            # lockstep, so every extra key is stall charged to all of them.
            page_exists = self._page_exists(page_keys[: valid_pages[-1]], transfer)
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

    def _restore_rate(self) -> float:
        """Cumulative restores per second, the signal behind the split gate.

        Not the load wait: splitting is what shortens the wait, so gating on it
        would oscillate. The restore rate is invariant to the split yet still
        separates the working points where splitting pays from those where it
        does not.

        Cumulative rather than trailing, because this is reached only while
        loads are pending: a trailing window would sample the bursts and never
        the quiet stretches between them.
        """
        elapsed = time.monotonic() - self._split_rate_start
        if elapsed < self._split_rate_min_seconds:
            return float("inf")  # too early to judge
        return self._split_loads_seen / elapsed

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
        if self._split_load:
            self._split_loads_seen += 1
            self._pending_pages[rid] = self._lookup_pages.pop(rid, None)
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # The tree node is already visible in L1. Dropping its transfer would
        # leave a device hit pointing at slots that were never populated.
        return False

    def num_completed_loads(self) -> int:
        return self._completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self._completed_loads.get_nowait()

    def start_layer_wise_loading(self) -> int:
        # Even a rank whose COMMIT discarded every transfer must vote.
        share = self._prepare_split_share() if self._split_load else None
        self._pending_pages.clear()
        if not self._pending:
            return -1
        self._freeze_gc_once()
        pending = self._pending
        rids = list(pending)
        plans = self._build_load_plans(list(pending.values()), share=share)
        ready_event = device_module.Event()
        ready_event.record()
        counter_index = self.layer_done_counter.update_producer()
        if share is not None:
            self._split_state[counter_index] = _SplitLoad(
                {plan.name: plan for plan in plans if plan.all_locations is not None}
            )
        self._load_queue.put((counter_index, rids, plans, ready_event))
        self._pending = {}
        self._stats["load"] += len(pending)
        return counter_index

    def _prepare_split_share(self) -> _SplitShare | None:
        agreed = self._split_agreement
        agreed.zero_()
        page_lists: list[list[str]] = []
        positions: list[list[int]] = []
        mask = None
        try:
            if not 0 < len(self._pending) <= self._split_max_rids:
                raise ValueError("empty or oversized split batch")
            # Rank-local, but it feeds the existing agreement, so one rank's
            # veto makes every rank skip: no protocol change, no new collective.
            if self._split_min_rate > 0:
                rate = self._restore_rate()
                # Recorded so a run shows what the gate saw, not just what it did.
                self._stats["split_rate_milli"] = int(min(rate, 1e6) * 1000)
                if rate < self._split_min_rate:
                    self._stats["split_rate_gated"] += 1
                    raise ValueError("restore rate below the split gate")
            total = 0
            for index, (rid, transfers) in enumerate(self._pending.items()):
                lookup_keys = self._pending_pages[rid]
                if lookup_keys is None:
                    raise ValueError("split lookup metadata was evicted")
                n = len(lookup_keys)
                if total + n > SPLIT_MAX_PAGES:
                    raise ValueError("split page mask exceeds its capacity")
                page_map = {key: index for index, key in enumerate(lookup_keys)}
                # Each logical source fans out to pools with the same keys.
                sources = {self.pools[t.name].indices_from_pool: t for t in transfers}
                for transfer in sources.values():
                    for key in transfer.keys:
                        page_map[key]  # Unknown keys veto the batch, on every rank.
                kv = sources.get(PoolName.KV)
                keys = list(kv.keys) if kv is not None else []
                page_lists.append(keys)
                positions.append([total + page_map[key] for key in keys])
                # Stable across Python processes; leave the sign bit clear so
                # both h and -h are representable in the agreement tensor.
                h = int.from_bytes(
                    blake2b(rid.encode(), digest_size=8).digest(), "little"
                )
                h &= (1 << 63) - 1
                slot = 1 + 4 * index
                agreed[slot], agreed[slot + 1] = h, -h
                agreed[slot + 2], agreed[slot + 3] = n, -n
                total += n
            # Allocate and fill before voting: a local preparation failure must
            # not leave peers entering the second collective without this rank.
            mask = torch.zeros(2 * total, dtype=torch.int32)
            for indices in positions:
                mask[indices] = 1
                mask[[total + index for index in indices]] = -1
            agreed[0] = 1
        except (KeyError, ValueError, MemoryError, RuntimeError):
            agreed[0] = 0

        torch.distributed.all_reduce(
            agreed, op=torch.distributed.ReduceOp.MIN, group=self._split_cpu_group
        )
        slots = agreed[1:].view(-1, 4)
        if not agreed[0].item() or not torch.equal(slots[:, ::2], -slots[:, 1::2]):
            if self._pending:
                self._stats["split_skipped_batches"] += 1
            return None
        torch.distributed.all_reduce(
            mask, op=torch.distributed.ReduceOp.MIN, group=self._split_cpu_group
        )
        common = mask[:total].nonzero().flatten().tolist()
        count = len(common)
        self._stats["split_divergent_pages"] += int(-mask[total:].sum()) - count
        self._stats["split_local_pages"] += sum(map(len, positions)) - count
        if count < self._split_min_pages or any(
            self._split_layout(count, group)[1] * self._split_world
            > self._split_recv.numel()
            for group in self._layer_groups()
        ):
            self._stats["split_skipped_batches"] += 1
            return None
        windows = _split_windows(count, self._split_world)
        start, end = windows[self._split_rank]
        owned = set(common[start:end])
        common = set(common)
        ordered = [
            sorted(zip(indices, keys)) for indices, keys in zip(positions, page_lists)
        ]
        self._stats["split_batches"] += 1
        return _SplitShare(
            [[key for index, key in pages if index in common] for pages in ordered],
            [{key for index, key in pages if index in owned} for pages in ordered],
            windows,
        )

    def _build_load_plans(
        self,
        request_transfers: list[list[PoolTransfer]],
        *,
        materialize_indices: Callable[[torch.Tensor], torch.Tensor] | None = None,
        share: _SplitShare | None = None,
    ) -> list[_PoolRangePlan]:
        """Build a batch plan shared by load and offload."""
        grouped: dict[PoolName, list[tuple[int, PoolTransfer]]] = {}
        for request_index, transfers in enumerate(request_transfers):
            for transfer in transfers:
                grouped.setdefault(transfer.name, []).append((request_index, transfer))

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
            common_locations = (
                []
                if share is not None and entry.indices_from_pool == PoolName.KV
                else None
            )
            for request_index, transfer in transfers:
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
                rows = entry.prepare_locations(prepared_indices)
                if len(rows) != len(page_keys):
                    raise ValueError(
                        f"UMBP pool {name} has different key and row counts."
                    )
                if common_locations is not None:
                    by_key = dict(zip(page_keys, rows))
                    common_keys = share.common[request_index]
                    common_locations.extend(by_key[key] for key in common_keys)
                    common_set = set(common_keys)
                    keep = [
                        index
                        for index, key in enumerate(page_keys)
                        if key not in common_set or key in share.owned[request_index]
                    ]
                    rows = [rows[index] for index in keep]
                    transfer_keys = [
                        key
                        for index in keep
                        for key in transfer_keys[
                            index * multiplier : (index + 1) * multiplier
                        ]
                    ]
                keys.extend(transfer_keys)
                locations.extend(rows)
            if len(keys) != len(locations) * entries_per_page:
                raise ValueError(
                    f"UMBP pool {name} plan mismatch: keys={len(keys)} "
                    f"rows={len(locations)} per_page={entries_per_page}."
                )
            plans.append(
                _PoolRangePlan(
                    name,
                    keys,
                    locations,
                    entries_per_page,
                    common_locations,
                    share.windows if common_locations is not None else (),
                )
            )

        if not plans or (share is None and not plans[0].keys):
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

    def _load_thread_func(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, rids, plans, ready_event = task
                try:
                    self._run_layer_wise_batch(counter_index, plans, ready_event)
                finally:
                    self._completed_loads.put(rids)
            finally:
                self._load_queue.task_done()

    def _all_layer_ranges(self, plan: _PoolRangePlan):
        """Every layer's ranges, accumulated per object.

        Offload requires one call to carry ranges that tile the object exactly,
        so an object's ranges must never be split across calls.

        The group is the pool's own layer stack, so it always covers the pool
        and the None return is unreachable; rejecting it keeps a future caller
        that passes a foreign plan from failing on a tuple unpack instead.
        """
        meta = self._layer_group_ranges(plan, self.pool_layers[plan.name])
        if meta is None:
            raise ValueError(
                f"UMBP pool {plan.name} covers none of its own layers "
                f"({self.pool_layers[plan.name]})."
            )
        return meta

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

    def _range_items(self, plan: _PoolRangePlan, layers: list[int]):
        """(base_ptr, row_stride, size, offset) per emitted range, in wire order.

        Grouped by layer, and within a layer by component. Every object emits
        this same tuple sequence; the only thing that varies across objects is
        the pointer, by ``row * row_stride``. That invariant is what the
        vectorized builder rests on.
        """
        entry = self.pools[plan.name]
        items: list[list[tuple[int, int, int, int]]] = []
        for logical_layer in layers:
            buffer_index = entry.layer_mapping.get(logical_layer)
            if buffer_index is None:
                continue
            items.append(
                [
                    (*component[buffer_index], offsets[buffer_index])
                    for component, offsets in zip(
                        entry.buffer_meta, entry._component_offsets
                    )
                ]
            )
        return items

    def _layer_group_ranges(self, plan: _PoolRangePlan, layers: list[int]):
        """One group of layers' ranges, one nested list per object.

        Built column-wise rather than object by object. A 256K restore emits
        ~63k ranges across its pools, and assembling those lists one page at a
        time cost ~43 ms per load -- serialized ahead of every group's transfer,
        on the same thread, so it landed directly on TTFT. Two redundancies pay
        for that: ``sizes`` and ``offsets`` do not depend on the row yet were
        rebuilt for every page, and ``ptrs`` is affine in the row so the whole
        column can be computed at once.

        Moving the work to a helper thread was tried and does not pay: the load
        thread has to re-acquire the GIL between blocking transfers and gives the
        saving straight back. It has to go away rather than move.

        Returns None when this pool covers none of the group, so the caller can
        skip the call.
        """
        items = self._range_items(plan, layers)
        if not items:
            return None
        rows = np.asarray(plan.locations, dtype=np.int64)
        if not rows.size:
            return [], [], []
        expected = len(rows) * plan.entries_per_page
        if expected != len(plan.keys):
            # One object per key, so a mismatch means pointers would be paired
            # with the wrong objects rather than anything failing loudly.
            raise ValueError(
                f"UMBP pool {plan.name} has {len(plan.keys)} keys for "
                f"{len(rows)} rows at {plan.entries_per_page} per page."
            )

        if self.pools[plan.name].packed:
            # One object per page, its ranges running (layer, component).
            flat = [item for layer_items in items for item in layer_items]
            base = np.fromiter((item[0] for item in flat), np.int64, len(flat))
            stride = np.fromiter((item[1] for item in flat), np.int64, len(flat))
            ptrs = (rows[:, None] * stride[None, :] + base[None, :]).tolist()
            # One list shared by every object instead of a copy each: the client
            # only reads these.
            sizes = [item[2] for item in flat]
            offsets = [item[3] for item in flat]
            return ptrs, [sizes] * len(rows), [offsets] * len(rows)

        # One object per (page, component), its ranges running over the layers.
        components = len(items[0])
        base = np.array([[i[0] for i in layer] for layer in items], np.int64).T
        stride = np.array([[i[1] for i in layer] for layer in items], np.int64).T
        ptrs = (
            (rows[:, None, None] * stride[None, :, :] + base[None, :, :])
            .reshape(len(rows) * components, -1)
            .tolist()
        )
        sizes = [[layer[index][2] for layer in items] for index in range(components)]
        offsets = [[layer[index][3] for layer in items] for index in range(components)]
        return ptrs, sizes * len(rows), offsets * len(rows)

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
        self, counter_index: int, plans: list[_PoolRangePlan], ready_event: object
    ) -> None:
        released = 0
        try:
            ready_event.synchronize()
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
                released = group[-1] + 1
        except BaseException as error:
            state = self._split_state.get(counter_index)
            if state is None:
                self.layer_done_counter.fail(counter_index, error)
            else:
                # Let forward reach the next group vote even if this rank's
                # read failed; failing its Future would bypass that collective.
                state.failure = error
                for layer in range(released, self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
            logger.exception("UMBP layer-wise load batch failed")

    def _layer_groups(self) -> list[list[int]]:
        return [
            list(range(start, min(start + self.layer_group, self.num_layers)))
            for start in range(0, self.num_layers, self.layer_group)
        ]

    def _split_layout(self, pages: int, group: list[int]):
        layout = []
        offset = 0
        width = -(-pages // self._split_world)
        # Static pool order, independent of rank-local COMMIT de-duplication.
        for name in self._split_pools:
            entry = self.pools[name]
            for layer in group:
                index = entry.layer_mapping.get(layer)
                if index is None:
                    continue
                for component, meta in zip(entry.components, entry.buffer_meta):
                    _, stride, size = meta[index]
                    layout.append((name, component[index], size // stride, offset))
                    offset += -(-(width * size) // 256) * 256
        return layout, offset

    def _split_rows(
        self, state: _SplitLoad, name: PoolName, rank: int, span: int, device
    ):
        key = (name, rank)
        if key not in state.rows:
            plan = state.plans[name]
            start, end = plan.windows[rank]
            rows = torch.tensor(
                plan.all_locations[start:end], dtype=torch.int64, device=device
            )
            if span > 1:
                rows = (rows[:, None] + torch.arange(span, device=device)).reshape(-1)
            state.rows[key] = rows
        return state.rows[key]

    @staticmethod
    def _split_view(buffer: torch.Tensor, offset: int, rows: int, like: torch.Tensor):
        size = rows * like.stride(0) * like.element_size()
        return (
            buffer[offset : offset + size].view(like.dtype).view(rows, *like.shape[1:])
        )

    def _exchange_ready_groups(self, counter_index: int, layer: int) -> None:
        state = self._split_state.get(counter_index)
        if state is None:
            return
        groups = self._layer_groups()
        target = layer // self.layer_group
        try:
            while state.exchanged < target:
                self._exchange_group(state, groups[state.exchanged + 1])
                state.exchanged += 1
            if target == len(groups) - 1 and state.status is not None:
                # The one host read per batch. A rank that failed at group k is
                # reported here rather than at k, so a few more layers run
                # against KV that is discarded anyway; nothing can hang,
                # because the allgather is issued unconditionally.
                if not state.status.item():
                    raise RuntimeError(
                        "UMBP split load failed on at least one rank."
                    ) from state.failure
        finally:
            if target == len(groups) - 1:
                self._split_state.pop(counter_index, None)

    def _exchange_group(self, state: _SplitLoad, group: list[int]) -> None:
        failure = state.failure
        # Deliberately outside the failure guard: `pages` comes from the plan
        # agreed at prepare time, so every rank derives the same layout whether
        # or not its own load succeeded. A rank returning early here would
        # strand its peers in the allgather below. Anything raised here is
        # symmetric, so it still propagates before any collective.
        if self._split_recv.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("UMBP split load cannot run inside CUDA graph capture.")
        pages = len(next(iter(state.plans.values())).all_locations)
        layout, slot_bytes = self._split_layout(pages, group)
        if failure is None:
            try:
                # Prepare *all* ranks' scatter indices before voting. Nothing
                # that can fail locally may strand peers in the allgather.
                for name, tensor, span, _ in layout:
                    for rank in range(self._split_world):
                        self._split_rows(state, name, rank, span, tensor.device)
                base = self._split_rank * slot_bytes
                for name, tensor, _, offset in layout:
                    rows = state.rows[name, self._split_rank]
                    if rows.numel():
                        torch.index_select(
                            tensor,
                            0,
                            rows,
                            out=self._split_view(
                                self._split_recv, base + offset, rows.numel(), tensor
                            ),
                        )
            except BaseException as error:
                failure = error

        # The vote still runs every group -- it is what keeps a locally failed
        # rank from stranding its peers -- but its result is read once, at the
        # end of the batch, instead of draining the stream per group.
        self._split_status.fill_(int(failure is None))
        torch.distributed.all_reduce(
            self._split_status, op=torch.distributed.ReduceOp.MIN, group=self._split_pg
        )
        if state.status is None:
            state.status = self._split_status.clone()
        else:
            torch.minimum(state.status, self._split_status, out=state.status)
        if failure is not None and state.failure is None:
            # Keep the first local cause so the deferred raise can report it.
            state.failure = failure
        if not layout:
            return
        base = self._split_rank * slot_bytes
        torch.distributed.all_gather_into_tensor(
            self._split_recv[: slot_bytes * self._split_world],
            self._split_recv[base : base + slot_bytes],
            group=self._split_pg,
        )
        if failure is not None:
            # Our rows are untrustworthy; the deferred check fails the batch.
            return
        for rank in range(self._split_world):
            if rank == self._split_rank:
                continue
            for name, tensor, _, offset in layout:
                rows = state.rows[name, rank]
                if rows.numel():
                    tensor.index_copy_(
                        0,
                        rows,
                        self._split_view(
                            self._split_recv,
                            rank * slot_bytes + offset,
                            rows.numel(),
                            tensor,
                        ),
                    )

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self._freeze_gc_once()
        ready_event = device_module.Event()
        ready_event.record()
        self._offload_queue.put((expanded, ready_event))
        return True

    def _take_offload_batch(self) -> tuple[list[_OffloadTask], bool]:
        """Block for one task, then take whatever else is already queued.

        Returns the tasks and whether the stop sentinel came with them; each
        item taken here needs one ``task_done()`` from the caller. Taking only
        what is already queued keeps a task from waiting on an unsubmitted peer.
        """
        first = self._offload_queue.get()
        if first is None:
            return [], True
        tasks = [first]
        pages = _offload_task_pages(first[0])
        while pages < self._offload_coalesce_pages:
            try:
                task = self._offload_queue.get_nowait()
            except Empty:
                break
            if task is None:
                return tasks, True
            tasks.append(task)
            pages += _offload_task_pages(task[0])
        return tasks, False

    def _offload_thread_func(self) -> None:
        while True:
            tasks, stopping = self._take_offload_batch()
            taken = len(tasks) + int(stopping)
            try:
                if tasks:
                    self._offload_batch(tasks)
            finally:
                for _ in range(taken):
                    self._offload_queue.task_done()
            if stopping:
                return

    def _offload_batch(self, tasks: list[_OffloadTask]) -> None:
        success = False
        try:
            success = self._run_offload(tasks)
        except BaseException:
            logger.exception("UMBP offload failed")
            success = False
        finally:
            # One result per task in submission order: the tree pairs them
            # positionally. A batch resolves as a unit because the first failed
            # pool stops the rest, leaving every task in it incomplete.
            for _ in tasks:
                self._offload_results.put(success)

    def _run_offload(self, tasks: list[_OffloadTask]) -> bool:
        for _, ready_event in tasks:
            ready_event.synchronize()
        next_slot = 0

        def materialize_indices(indices: torch.Tensor) -> torch.Tensor:
            nonlocal next_slot
            slot = next_slot
            next_slot += 1
            return self._materialize_offload_indices(indices, slot)

        plans = self._build_load_plans(
            [expanded for expanded, _ in tasks],
            materialize_indices=materialize_indices,
        )
        # Over plans, not transfers: a plan already carries every task's keys
        # for its pool, so walking transfers would put that pool once per task.
        for plan in plans:
            entry = self.pools[plan.name]
            # From the pool layout, never from the ranges below: see
            # _object_sizes_per_page.
            per_page = _object_sizes_per_page(entry)
            if len(per_page) != plan.entries_per_page:
                raise ValueError(
                    f"UMBP pool {plan.name} declares {len(per_page)} object "
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
                        plan.name,
                        start,
                        min(end, len(plan.keys)),
                        sum(bool(value) for value in results),
                        len(chunk_keys),
                        len(results),
                    )
                    return False

        self._stats["offload"] += len(tasks)
        self._stats["offload_batches"] += 1
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
        return self._offload_results.get_nowait()

    def reset(self) -> None:
        self._pending.clear()
        self._load_queue.join()
        self._offload_queue.join()
        self._lookup_pages.clear()
        self._pending_pages.clear()
        self._split_state.clear()
        while True:
            try:
                self._offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self._completed_loads.get_nowait()
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
        if self._standalone_process_mode and self._registered:
            # StandaloneProcess deregistration is client-wide; one call tears
            # down every registered region. Keep the GPU tensors alive until
            # the synchronous RPC has completed successfully.
            self.storage.client.deregister_memory(self._registered[0][0])
        logger.info("UMBP direct linker stats: %s", self._stats)
        self.storage.close()
        self._closed = True
