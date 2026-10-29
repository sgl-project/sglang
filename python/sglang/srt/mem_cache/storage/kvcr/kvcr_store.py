# SPDX-License-Identifier: Apache-2.0
"""Host-staged KVCR L3: SGLang HBM -> HiCache host -> KVCR DRAM.

Reads deliver directly into the host slots owned/locked by HiCache. KVCR
serves only its own deposited DRAM, never unpinned framework host pages.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import mmap
import os
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from queue import Empty, SimpleQueue

from sglang.srt.managers.kv_hints import decode_kv_hints_envelope
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.storage.kvcr.endpoints import (
    control_address,
    rank_endpoint,
    rank_offset,
)
from sglang.srt.mem_cache.storage.kvcr.hints import peer_hint
from sglang.srt.mem_cache.storage.kvcr.host_layout import (
    StorageKeyAdapter,
    block_key,
    describe_pool,
    page_slots,
    resume_boundaries,
    storage_namespace,
)
from sglang.srt.mem_cache.storage.kvcr.inventory import KVCRInventory

logger = logging.getLogger(__name__)

# Pools HiCache restores as TRAILING_PAGES windows (SWA, its DeepSeek V4
# compressor states, Mamba): a page need not have state in them.
_TRAILING_POOLS = frozenset(
    str(name)
    for name in (
        PoolName.SWA,
        PoolName.DRAFT_SWA,
        PoolName.MAMBA,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        PoolName.DEEPSEEK_V4_C128_STATE,
    )
)


def _bool_option(options: dict, name: str) -> bool:
    value = options.get(name, False)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _allocate_dram_pool(length):
    """Keep KVCR's private DRAM pool off THP without changing worker-wide THP."""
    if not hasattr(mmap, "MAP_PRIVATE") or not hasattr(mmap, "MAP_ANONYMOUS"):
        return mmap.mmap(-1, length)
    allocation = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if hasattr(allocation, "madvise") and hasattr(mmap, "MADV_NOHUGEPAGE"):
        try:
            allocation.madvise(mmap.MADV_NOHUGEPAGE)
        except OSError:
            logger.debug("KVCR DRAM ordinary-page advice unavailable", exc_info=True)
    return allocation


class _NoFrameworkPinning:
    """Answer misses when a peer asks for a non-KVCR-owned source."""

    def __init__(self):
        self._results = SimpleQueue()
        self._next_id = 0

    def request_pin(self, keys):
        self._next_id += 1
        self._results.put((self._next_id, None))
        return self._next_id

    def poll_pin_results(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                return results

    def release_pin(self, handle):
        return False


class KVCRStore(HiCacheStorage):
    """Optional NIXL/UCX L3 backend using KVCR's indexed-region public API."""

    requires_lookup_confirmation = True

    def __init__(self, storage_config, mem_pool_host):
        self.config = storage_config
        self.mem_pool_host = mem_pool_host
        self.registered_pools = {}
        self._layouts = {}
        self._runner = None
        self._maps = []
        self._lock = threading.RLock()
        self._completion = threading.Condition(self._lock)
        self._completed = {}
        # scope -> physical trailing-state keys and their local-DRAM claims.
        # A router FETCHABLE verdict alone does not prove source residency.
        self._lookup_claims = {}
        self._poll_thread = None
        self._stop_poll = threading.Event()
        self._closed = False
        self._pinning = _NoFrameworkPinning()
        self._extra = dict(storage_config.extra_config or {})
        self.publishes_kv_events = _bool_option(self._extra, "publish_inventory")
        if self.publishes_kv_events and (
            self._extra.get("enable_remote_hint", True) is not True
            or self._extra.get("control_advertise_host") is None
        ):
            raise ValueError(
                "KVCR inventory requires enabled remote hints and an explicit "
                "control_advertise_host"
            )
        self._inventory = KVCRInventory() if self.publishes_kv_events else None
        if "startup_timeout_seconds" in self._extra and (
            type(self._extra["startup_timeout_seconds"]) is not int
            or self._extra["startup_timeout_seconds"] <= 0
        ):
            raise ValueError("startup_timeout_seconds must be a positive integer")
        self.prefetch_batch_pages = self._extra.get(
            "prefetch_batch_pages", STORAGE_BATCH_SIZE
        )
        if type(self.prefetch_batch_pages) is not int or self.prefetch_batch_pages <= 0:
            raise ValueError("prefetch_batch_pages must be a positive integer")
        self._metrics = None
        if storage_config.enable_storage_metrics:
            from sglang.srt.observability.metrics_collector import StorageMetrics

            self._metrics = StorageMetrics()
        self._verify_transfer_bytes = _bool_option(self._extra, "verify_transfer_bytes")
        self._align_storage_prefix = _bool_option(self._extra, "align_storage_prefix")
        self._diagnostic_telemetry = _bool_option(self._extra, "diagnostic_telemetry")
        self.supports_combined_page_reads = _bool_option(
            self._extra, "combine_page_reads"
        )
        self.supports_combined_page_writes = _bool_option(
            self._extra, "combine_page_writes"
        )
        self._diagnostic_epoch = time.monotonic()
        self._offset = rank_offset(storage_config)
        self._share_replicated_mla = _bool_option(self._extra, "share_replicated_mla")
        if self._share_replicated_mla and (
            not storage_config.is_mla_model
            or storage_config.pp_size != 1
            or storage_config.attn_cp_size != 1
            or storage_config.should_split_heads
            or mem_pool_host.kv_buffer is None
        ):
            raise ValueError(
                "Shared MLA storage requires replicated physical KV, CP=1 and PP=1"
            )
        # Shared mode is an explicit claim that KV/indexer bytes are identical
        # across attention TP ranks. Never apply it to sharded hybrid state.
        self.requires_rank_local_backup = not self._share_replicated_mla
        self._source_offset = 0 if self._share_replicated_mla else self._offset
        self._agent_name = f"sglang-kvcr-{uuid.uuid4().hex}"
        self.kv_event_shard = self._offset
        self.kv_event_shard_count = (
            1
            if self._share_replicated_mla
            else storage_config.tp_size * storage_config.attn_cp_size
        )
        if storage_config.should_split_heads:
            raise ValueError("KVCR does not support heterogeneous TP head splitting")
        if storage_config.pp_size != 1:
            raise ValueError("KVCR HiCacheStorage currently requires PP=1")
        self.register_mem_pool_host(mem_pool_host)

    def register_mem_pool_host(self, mem_pool_host):
        self.mem_pool_host = mem_pool_host
        # Hybrid logical anchors have no payload. The real pools arrive via v2.
        if mem_pool_host.kv_buffer is not None:
            self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)

    def get_kv_hint_source(self):
        if (
            self._offset != 0
            or self._extra.get("enable_remote_hint", True) is not True
            or self._extra.get("control_advertise_host") is None
        ):
            return None
        host, port = control_address(self.config, self._extra)
        source = {
            "dp_rank": self.config.dp_rank,
            "capabilities": ["router_hint"],
            "source_control_endpoint": f"tcp://{host}:{port}",
        }
        if self.publishes_kv_events:
            namespace = storage_namespace(
                self.config, self._layouts, shared_mla=self._share_replicated_mla
            )
            source["capabilities"].append("kvcr_inventory")
            # This adapter's pool is process-local: a cold engine must get a new
            # owner, never inherit a previous engine's retained routing state.
            source["cache_owner_id"] = (
                f"{namespace[:32]}:{namespace[32:]}/0000000000000000/"
                f"{self._agent_name.rsplit('-', 1)[-1]}"
            )
        return source

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        name = str(host_pool_name)
        with self._lock:
            if self._share_replicated_mla and name not in ("kv", "indexer"):
                raise ValueError(
                    "Shared MLA storage only supports replicated KV/indexer pools"
                )
            if name in self.registered_pools:
                if self.registered_pools[name] is not host_pool:
                    raise ValueError(f"Host pool {name} was registered twice")
                return
            if name == str(PoolName.KV) and host_pool.kv_buffer is None:
                return
            if self._runner is not None:
                raise ValueError(
                    "KVCR host pools must be registered before the first IO"
                )
            self._layouts[name] = describe_pool(name, host_pool)
            self.registered_pools[name] = host_pool

    def _start(self):
        if self._closed:
            raise RuntimeError("KVCR storage backend is closed")
        if self._runner is not None:
            return
        from kvcr import KVCR, KVCRBindings, ZmqPeerControlChannel
        from kvcr.config import KVCRBackendConfigs, KVCRConfig, LocalDramOptions
        from kvcr.types import KVCRStartupError, RegionDescriptor

        from sglang.srt.mem_cache.storage.kvcr.diagnostics import DiagnosticStats

        if not self._layouts:
            raise ValueError("KVCR has no registered physical host pools")
        self._namespace = storage_namespace(
            self.config, self._layouts, shared_mla=self._share_replicated_mla
        )
        if self._inventory is not None:
            self._inventory.configure(
                self._namespace,
                [name for name in self._layouts if name not in _TRAILING_POOLS],
            )
        total = sum(
            component.size * component.count
            for layout in self._layouts.values()
            for component in layout.components
        )
        budget = self._extra.get("local_dram_bytes", 8 << 30)
        pages_by_pool = self._extra.get("local_dram_pool_pages", {})
        if (
            type(budget) is not int
            or budget <= 0
            or not isinstance(pages_by_pool, dict)
        ):
            raise ValueError("Invalid KVCR local DRAM budget")
        host, port = control_address(self.config, self._extra)
        dp_stride = self.config.tp_size * self.config.attn_cp_size * self.config.pp_size
        offset = self.config.dp_rank * dp_stride + self._offset
        base_port = port - offset
        nixl_base_port = self._extra.get("nixl_listen_port", base_port + 1024)
        if type(nixl_base_port) is not int or not 0 < nixl_base_port <= 65535:
            raise ValueError("KVCR nixl_listen_port must be a valid integer port")
        nixl_port = nixl_base_port + offset
        if not 0 < port <= 65535 or not 0 < nixl_port <= 65535 or port == nixl_port:
            raise ValueError("KVCR control/NIXL port range is invalid or overlaps")
        layouts, regions, plans = [], [], []
        for name, layout in self._layouts.items():
            pages = pages_by_pool.get(
                name, budget * layout.components[0].count // total
            )
            if type(pages) is not int or pages < 1:
                raise ValueError(f"KVCR DRAM budget cannot hold one page of {name}")
            for component in layout.components:
                length = pages * component.size
                if not self._share_replicated_mla or self.config.tp_rank == 0:
                    plans.append((component.name, length))
                layouts.append((component.name, component.size))
                regions.append(
                    RegionDescriptor(
                        addr=component.address,
                        size=component.size,
                        stride=component.stride,
                        count=component.count,
                        label=component.name,
                    )
                )
        channel = ZmqPeerControlChannel(
            bind_host=self._extra.get("control_host", "0.0.0.0"),
            bind_port=port,
            advertise_host=host,
        )
        self._control_endpoint = channel.endpoint
        self._owner_endpoint = rank_endpoint(channel.endpoint, -self._offset)
        if self._owner_endpoint is None:
            channel.close()
            raise ValueError("Cannot identify the replica's TP0 control endpoint")
        allocations, mappings = [], []
        with ExitStack() as resources:
            resources.callback(channel.close)
            for name, length in plans:
                allocation = _allocate_dram_pool(length)
                resources.callback(allocation.close)
                mappings.append(allocation)
                address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
                allocations.append((name, address, length))
            try:
                # Do not pass the new option when unset: ordinary deployments
                # remain compatible with KVCR versions predating this field.
                startup_options = (
                    {"startup_timeout_seconds": self._extra["startup_timeout_seconds"]}
                    if "startup_timeout_seconds" in self._extra
                    else {}
                )
                self._runner = KVCR(
                    config=KVCRConfig(
                        nixl_agent_name=self._agent_name,
                        nixl_listen_port=nixl_port,
                        pool_layouts=layouts,
                        enable_telemetry=self._diagnostic_telemetry,
                        operation_timeout_ms=self._extra.get(
                            "operation_timeout_ms", 30000
                        ),
                        abandon_timeout_ms=self._extra.get("abandon_timeout_ms", 60000),
                        **startup_options,
                    ),
                    bindings=KVCRBindings(
                        request_pin=self._pinning.request_pin,
                        poll_pin_results=self._pinning.poll_pin_results,
                        release_pin=self._pinning.release_pin,
                        framework_control=channel,
                        key_adapter=StorageKeyAdapter(),
                        on_resilience_event=self._resilience_event,
                        stats_factory=(
                            DiagnosticStats if self._diagnostic_telemetry else None
                        ),
                        inventory_sink=(
                            self._record_inventory
                            if self.publishes_kv_events or self._diagnostic_telemetry
                            else None
                        ),
                    ),
                    backend_configs=KVCRBackendConfigs(
                        framework_regions=regions,
                        local_dram=(
                            LocalDramOptions(pools=allocations) if allocations else None
                        ),
                    ),
                )
            except KVCRStartupError:
                # Native work still owns these mappings. They cannot be reclaimed.
                resources.pop_all()
                logger.critical(
                    "Nonquiescent KVCR startup; terminating worker", exc_info=True
                )
                os._exit(1)
            except Exception:
                # A dead prefetch thread otherwise strands wait_complete while
                # unrelated HTTP health checks continue to report success.
                logger.critical(
                    "KVCR startup failed; terminating worker", exc_info=True
                )
                os._exit(1)
            self._maps = mappings
            resources.pop_all()
        # Peer source operations also need the public main-side poll while this
        # worker is idle. Polling only from a local read/write strands source
        # pin/refusal and lease-release callbacks until another request arrives.
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="kvcr-storage-completions", daemon=True
        )
        self._poll_thread.start()
        logger.info(
            "KVCR L3 ready: endpoint=%s physical_pools=%s bytes=%d namespace=%s",
            channel.endpoint,
            layouts,
            sum(size for _, _, size in allocations),
            self._namespace,
        )

    @staticmethod
    def _resilience_event(error):
        from kvcr.types import TransferError

        if isinstance(error, TransferError) and error.state == "uncertain":
            # HiCache cannot quarantine arbitrary source/destination pages yet.
            # Returning a miss would let it reuse memory still exposed to NIXL.
            logger.critical("Unsafe KVCR transfer; terminating worker: %s", error)
            os._exit(1)
        logger.warning("KVCR resilience event: %s", error)

    def _operation_hint(self, extra_info, keys=None):
        hint = (
            peer_hint(extra_info, self._source_offset, keys)
            if self._extra.get("enable_remote_hint", True)
            else None
        )
        if hint is not None:
            return hint
        if not self._share_replicated_mla or self.config.tp_rank == 0 or not keys:
            return None
        # Readers without an external hint use their own replica's TP0. Their
        # query is advisory; TP0's real local query plus HiCache's MIN reduction
        # bounds the hit length. Final delivery results are also rank-reduced.
        return {
            "protocol_version": "0.1",
            "message_id": "replica-local",
            "actions": [
                {
                    "action_id": "replica-local",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": self._owner_endpoint,
                        "block_hashes": list(
                            dict.fromkeys(int(key[:16], 16) for key in keys)
                        ),
                        "mode": "copy",
                    },
                }
            ],
        }

    @contextmanager
    def _hint(self, extra_info, keys=None):
        request_id = None
        hint = self._operation_hint(extra_info, keys)
        if hint is not None:
            request_id = uuid.uuid4().hex
            self._runner.submit_hint(
                decode_kv_hints_envelope(hint), request_id=request_id
            )
        try:
            yield request_id
        finally:
            if request_id is not None:
                self._runner.discard_hint(request_id)

    def _wait(self, handle):
        # There can be concurrent backup and prefetch threads. One caller may
        # drain another caller's completion; keep it until the owner consumes it.
        with self._completion:
            while handle not in self._completed:
                self._completed.update(self._runner.poll_completed())
                if handle not in self._completed:
                    self._completion.wait(timeout=0.001)
            result = self._completed.pop(handle)
            self._completion.notify_all()
            return result

    def _poll_loop(self):
        while not self._stop_poll.is_set():
            try:
                with self._completion:
                    self._completed.update(self._runner.poll_completed())
                    self._completion.notify_all()
                    if self._diagnostic_telemetry:
                        self._drain_diagnostics()
            except Exception:
                # Native progress failure cannot safely be treated as a cache
                # miss: HiCache could recycle a buffer still exposed to NIXL.
                logger.critical(
                    "KVCR L3 progress failed; terminating worker", exc_info=True
                )
                os._exit(1)
            self._stop_poll.wait(0.001)

    def _record_inventory(self, event):
        if self._inventory is not None and event.tier.value == "DRAM":
            self._inventory.update(event)
        if self._diagnostic_telemetry:
            self._diagnostic_inventory(event)

    def pending_kv_events(self):
        with self._lock:
            return len(self._inventory.events) if self._inventory is not None else 0

    def take_kv_events(self):
        with self._lock:
            return self._inventory.take() if self._inventory is not None else []

    @contextmanager
    def _inventory_operation(self, extra_info, blocks):
        retained = set()
        if self._inventory is not None:
            extra = getattr(extra_info, "extra_info", None) or {}
            retained = self._inventory.register(
                extra.get("kv_event_metadata", {}), blocks
            )
        try:
            yield
        finally:
            if self._inventory is not None:
                self._inventory.release(retained)

    def _diagnostic_inventory(self, event):
        # Called on the public main side. Full storage keys permit exact offline
        # joins across peers; u64 router hints alone cannot distinguish pools.
        # Published removal proves absence then, not that every later miss was
        # an eviction. A later store event supersedes the removal.
        logger.info(
            "KVCR L3 inventory %s",
            json.dumps(
                {
                    "epoch": time.time(),
                    "agent": self._agent_name,
                    "endpoint": self._control_endpoint,
                    "namespace": self._namespace,
                    "tier": event.tier.value,
                    "removed": event.removed,
                    "keys": [key.decode() for key in event.keys],
                },
                separators=(",", ":"),
            ),
        )

    def _drain_diagnostics(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._diagnostic_epoch < 5:
            return
        stats = self._runner.get_stats()
        if stats is not None:
            logger.info(
                "KVCR L3 native stats %s",
                json.dumps(
                    {
                        "epoch": time.time(),
                        "agent": self._agent_name,
                        "endpoint": self._control_endpoint,
                        "interval_seconds": now - self._diagnostic_epoch,
                        "histogram_bounds_seconds": stats.BOUNDS,
                        "metrics": stats.reduce(),
                    },
                    separators=(",", ":"),
                ),
            )
        self._diagnostic_epoch = now

    def _transfer_digests(self, blocks, results):
        """Diagnostic-only hashes while HiCache still protects the host pages.

        Use after native completion, before returning the pages to the
        controller. This reads CPU buffers only, and covers every component.
        It does not prove the subsequent host-to-GPU copy or model numerics.
        """
        components = {
            component.name: component
            for layout in self._layouts.values()
            for component in layout.components
        }
        digests = {}
        for key, refs in blocks.items():
            if key not in results or not results[key].success:
                continue
            digest = hashlib.blake2b(digest_size=32)
            for ref in refs:
                component = components[ref.label]
                digest.update(ref.label.encode())
                digest.update(
                    ctypes.string_at(
                        component.address + ref.element_index * component.stride,
                        component.size,
                    )
                )
            digests[key.decode()] = digest.hexdigest()
        return digests

    def _io(self, transfers, *, is_set, extra_info):
        from kvcr.types import MemoryRef

        if is_set and self._share_replicated_mla and self.config.tp_rank != 0:
            raise RuntimeError("Replicated MLA L3 backup must run on TP0")
        diagnostic = self._diagnostic_telemetry
        entered = time.perf_counter() if diagnostic else None
        entered_epoch = time.time() if diagnostic else None
        with self._lock:
            acquired = time.perf_counter() if diagnostic else None
            self._start()
            blocks, groups = {}, {}
            for transfer in transfers:
                name = str(transfer.name)
                keys = transfer.keys or []
                if name == str(PoolName.KV) and name not in self._layouts:
                    groups[name] = [(None, True) for _ in keys]
                    continue
                layout = self._layouts[name]
                slots = page_slots(layout, keys, transfer.host_indices)
                groups.setdefault(name, [])
                for key, slot in zip(keys, slots, strict=True):
                    identity = block_key(key, self._namespace, name)
                    refs = [
                        MemoryRef(
                            end_point_name=self._agent_name,
                            label=component.name,
                            element_index=slot,
                        )
                        for component in layout.components
                    ]
                    if identity in blocks:
                        raise ValueError("Duplicate pool/page in one KVCR operation")
                    blocks[identity] = refs
                    groups[name].append((identity, False))
            if not blocks:
                return {name: [True] * len(group) for name, group in groups.items()}
            described = time.perf_counter() if diagnostic else None
            started = time.perf_counter() if self._metrics is not None else None
            sequence = None
            if is_set and self._align_storage_prefix:
                # HiCache reuse may not touch KVCR's LRU. Refresh parents before
                # allocation/eviction and again after deposit for tail-first order.
                # This does not claim absent or non-ready blocks. Parent keys
                # require hicache_storage_pass_prefix_keys.
                prefix = getattr(extra_info, "prefix_keys", None) or []
                sequence = [
                    block_key(key, self._namespace, str(transfer.name))
                    for transfer in transfers
                    if str(transfer.name) in self._layouts
                    for key in [*prefix, *(transfer.keys or [])]
                ]
                self._runner.align_sequence(sequence, use_current_time=True)
            aligned = time.perf_counter() if diagnostic else None
            with (
                self._inventory_operation(extra_info, blocks),
                self._hint(extra_info, keys=None if is_set else blocks) as request_id,
            ):
                prepared = time.perf_counter() if diagnostic else None
                handle = (
                    self._runner.deposit(blocks)
                    if is_set
                    else self._runner.deliver(blocks, request_id=request_id)
                )
                submitted = time.perf_counter() if diagnostic else None
                # _wait releases the condition's recursive lock while sleeping.
                result = self._wait(handle)
                finished = time.perf_counter() if diagnostic else None
                if diagnostic:
                    logger.info(
                        "KVCR L3 operation %s",
                        json.dumps(
                            {
                                "epoch": time.time(),
                                "start_epoch": entered_epoch,
                                "agent": self._agent_name,
                                "endpoint": self._control_endpoint,
                                "operation": handle,
                                "stage": "primary",
                                "direction": "deposit" if is_set else "deliver",
                                "source": (
                                    self._operation_hint(extra_info, blocks)["actions"][
                                        0
                                    ]["payload"]["source_control_endpoint"]
                                    if request_id is not None
                                    else None
                                ),
                                "lock_wait_ms": (acquired - entered) * 1000,
                                "prepare_ms": (prepared - acquired) * 1000,
                                "descriptor_ms": (described - acquired) * 1000,
                                "align_before_ms": (aligned - described) * 1000,
                                "hint_ms": (prepared - aligned) * 1000,
                                "submit_ms": (submitted - prepared) * 1000,
                                "wait_ms": (finished - submitted) * 1000,
                                "requested_blocks": len(blocks),
                                "requested_spans": sum(map(len, blocks.values())),
                                "pools": [str(t.name) for t in transfers],
                                "failed_keys": [
                                    key.decode()
                                    for key in blocks
                                    if key not in result or not result[key].success
                                ],
                            },
                            separators=(",", ":"),
                        ),
                    )
                if (
                    not is_set
                    and self._share_replicated_mla
                    and self.config.tp_rank != 0
                    and request_id is not None
                ):
                    failed = {
                        key: refs
                        for key, refs in blocks.items()
                        if key not in result or not result[key].success
                    }
                    if (
                        failed
                        and self._operation_hint(extra_info, blocks)["actions"][0][
                            "payload"
                        ]["source_control_endpoint"]
                        != self._owner_endpoint
                    ):
                        # An external hint must not hide this replica's own
                        # TP0-resident pages. Retry only terminal misses, under
                        # a fresh scope, while HiCache still owns every slot.
                        # Successful external pages are never written again;
                        # uncertain native transfers still terminate the worker.
                        fallback_epoch = time.time() if diagnostic else None
                        fallback_started = time.perf_counter() if diagnostic else None
                        with self._hint(None, failed) as owner_request_id:
                            fallback_prepared = (
                                time.perf_counter() if diagnostic else None
                            )
                            fallback_handle = self._runner.deliver(
                                failed, request_id=owner_request_id
                            )
                            fallback_submitted = (
                                time.perf_counter() if diagnostic else None
                            )
                            recovered = self._wait(fallback_handle)
                            fallback_finished = (
                                time.perf_counter() if diagnostic else None
                            )
                        if diagnostic:
                            logger.info(
                                "KVCR L3 operation %s",
                                json.dumps(
                                    {
                                        "epoch": time.time(),
                                        "start_epoch": fallback_epoch,
                                        "agent": self._agent_name,
                                        "endpoint": self._control_endpoint,
                                        "operation": fallback_handle,
                                        "parent_operation": handle,
                                        "stage": "replica-local-fallback",
                                        "direction": "deliver",
                                        "source": self._owner_endpoint,
                                        "lock_wait_ms": 0,
                                        "prepare_ms": (
                                            fallback_prepared - fallback_started
                                        )
                                        * 1000,
                                        "submit_ms": (
                                            fallback_submitted - fallback_prepared
                                        )
                                        * 1000,
                                        "wait_ms": (
                                            fallback_finished - fallback_submitted
                                        )
                                        * 1000,
                                        "requested_blocks": len(failed),
                                        "requested_spans": sum(
                                            map(len, failed.values())
                                        ),
                                        "pools": [str(t.name) for t in transfers],
                                        "failed_keys": [
                                            key.decode()
                                            for key in failed
                                            if key not in recovered
                                            or not recovered[key].success
                                        ],
                                    },
                                    separators=(",", ":"),
                                ),
                            )
                        result = {
                            **result,
                            **{
                                key: recovered[key]
                                for key in failed
                                if key in recovered
                            },
                        }
                if sequence is not None:
                    self._runner.align_sequence(sequence, use_current_time=True)
                if self._verify_transfer_bytes:
                    logger.info(
                        "KVCR L3 byte verification %s",
                        json.dumps(
                            {
                                "direction": "deposit" if is_set else "deliver",
                                "namespace": self._namespace,
                                "digests": self._transfer_digests(blocks, result),
                            },
                            sort_keys=True,
                        ),
                    )
            if started is not None:
                self._record_metrics(
                    transfers,
                    blocks,
                    result,
                    elapsed=time.perf_counter() - started,
                    is_set=is_set,
                )
        return {
            name: [
                logical or (key in result and result[key].success)
                for key, logical in group
            ]
            for name, group in groups.items()
        }

    def _record_metrics(self, transfers, blocks, results, *, elapsed, is_set):
        """Count completed logical pages once, but every transferred byte.

        A primary page with a failed requested sidecar is not a usable page.
        Its successfully moved primary bytes still consumed transport bandwidth.
        HiCache separately accounts for admission and resume-boundary validity.
        Called with the IO lock held, after terminal native completion.
        """
        components = {
            component.name: component
            for layout in self._layouts.values()
            for component in layout.components
        }
        page_results, transferred_bytes = {}, 0
        for identity, refs in blocks.items():
            successful = identity in results and results[identity].success
            key = identity.split(b"#", 1)[0].decode()
            page_results.setdefault(key, []).append(successful)
            if successful:
                transferred_bytes += sum(components[ref.label].size for ref in refs)
        primary_keys = {
            key
            for transfer in transfers
            if str(transfer.name) == str(PoolName.KV)
            for key in (transfer.keys or [])
        }
        pages = sum(all(page_results.get(key, [False])) for key in primary_keys)
        bandwidth = transferred_bytes / max(elapsed, 1e-9) / 1e9
        if is_set:
            self._metrics.backup_pgs.append(pages)
            self._metrics.backup_bandwidth.append(bandwidth)
        else:
            self._metrics.prefetch_pgs.append(pages)
            self._metrics.prefetch_bandwidth.append(bandwidth)

    def get_stats(self):
        """Atomically hand HiCache its standard interval snapshot."""
        with self._lock:
            if self._metrics is None:
                return None
            from sglang.srt.observability.metrics_collector import StorageMetrics

            snapshot, self._metrics = self._metrics, StorageMetrics()
            return snapshot

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        return self.batch_get_v2(
            [PoolTransfer(PoolName.KV, host_indices=host_indices, keys=keys)],
            extra_info,
        )[str(PoolName.KV)]

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        return self.batch_set_v2(
            [PoolTransfer(PoolName.KV, host_indices=host_indices, keys=keys)],
            extra_info,
        )[str(PoolName.KV)]

    def batch_get_v2(self, transfers, extra_info=None):
        return self._io(transfers, is_set=False, extra_info=extra_info)

    def batch_set_v2(self, transfers, extra_info=None):
        return self._io(transfers, is_set=True, extra_info=extra_info)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        from kvcr.types import QueryStatus

        required = list(pool_transfers or [])
        if str(PoolName.KV) in self._layouts:
            required = [PoolTransfer(PoolName.KV)] + required
        elif not required:
            # A logical KV anchor has no payload to prove a page. Answer a
            # bare KV query from the pools that hold every stored page.
            required = [
                PoolTransfer(name)
                for name in self._layouts
                if name not in _TRAILING_POOLS
            ]
        if not required or not keys:
            # An empty intersection is not evidence of stored data. In
            # particular, trailing-only layouts cannot prove a bare prefix.
            return PoolTransferResult(0, {}, [])
        with self._lock:
            self._start()
            identities = [
                block_key(key, self._namespace, str(transfer.name))
                for transfer in required
                for key in keys
            ]
            with self._hint(extra_info, identities) as request_id:
                statuses = self._runner.query(identities, request_id=request_id)
        restorable, hits = set(range(1, len(keys) + 1)), {}
        for index, transfer in enumerate(required):
            status = statuses[index * len(keys) : (index + 1) * len(keys)]
            exists = [s in (QueryStatus.HIT, QueryStatus.FETCHABLE) for s, _ in status]
            trailing = None
            if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys or []))
            elif transfer.hit_policy != PoolHitPolicy.ALL_PAGES:
                raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")
            valid = resume_boundaries(exists, trailing=trailing)
            hits[str(transfer.name)] = max(valid, default=0)
            restorable.intersection_update(valid)
        valid = sorted(restorable)
        return PoolTransferResult(max(valid, default=0), hits, valid)

    def _release_lookup_locked(self, scope):
        claims = self._lookup_claims.get(scope)
        if not claims:
            self._lookup_claims.pop(scope, None)
            return
        handles = list(claims.values())
        released = self._runner.release(handles)
        if len(released) != len(handles) or not all(ok for _, ok in released):
            raise RuntimeError("KVCR trailing-state claim release failed")
        del self._lookup_claims[scope]

    def release_lookup(self, scope: str) -> None:
        with self._lock:
            self._release_lookup_locked(scope)

    def release_all_lookup_claims(self) -> None:
        with self._lock:
            for scope in list(self._lookup_claims):
                self._release_lookup_locked(scope)

    def confirm_lookup(
        self, keys, pool_transfers, hit_pages, extra_info, scope
    ) -> bool:
        """Physically fetch the selected trailing checkpoint and pin its copy.

        ``FETCHABLE`` means that a logical page matched a router hint; it is
        not a source inventory check. Fetch must finish successfully for every
        required trailing pool before this boundary can be admitted. The
        claims keep the fetched local copies resident until host delivery ends.
        """
        if not scope or not isinstance(scope, str):
            raise ValueError("KVCR lookup confirmation requires an operation scope")
        if not isinstance(hit_pages, int) or not 0 <= hit_pages <= len(keys):
            raise ValueError("Invalid KVCR lookup boundary")
        with self._lock:
            self._release_lookup_locked(scope)
            if hit_pages == 0:
                return False
            self._start()
            windows = {}
            for transfer in pool_transfers:
                if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                    continue
                name = str(transfer.name)
                window_size = max(1, len(transfer.keys or []))
                if window_size > hit_pages or name not in self._layouts:
                    return False
                windows[name] = [
                    block_key(key, self._namespace, name)
                    for key in keys[hit_pages - window_size : hit_pages]
                ]
            if not windows:
                return True
            claims = self._lookup_claims.setdefault(scope, {})
            try:
                for name, pending in windows.items():
                    labels = [part.name for part in self._layouts[name].components]
                    with self._hint(extra_info, pending) as request_id:
                        handle = self._runner.fetch(
                            pending,
                            request_id=request_id,
                            expected_layout=labels,
                        )
                        result = self._wait(handle)
                    complete = True
                    for key in pending:
                        entry = result.get(key)
                        if entry is None or not entry.success:
                            complete = False
                        elif entry.release_handle is None:
                            complete = False
                        else:
                            claims[key] = entry.release_handle
                    if not complete:
                        self._release_lookup_locked(scope)
                        return False
                return True
            except Exception:
                self._release_lookup_locked(scope)
                raise

    def batch_exists(self, keys, extra_info=None):
        return self.batch_exists_v2(keys, extra_info=extra_info).kv_hit_pages

    def exists(self, key):
        return bool(self.batch_exists([key]))

    # Tensor-valued v0 IO cannot identify registered, lifetime-protected pages.
    # The factory/controller explicitly select the indexed v1/v2 path instead.
    def get(self, key, target_location=None, target_sizes=None):
        raise NotImplementedError("KVCR requires indexed HiCache host pages")

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise NotImplementedError("KVCR requires indexed HiCache host pages")

    def set(self, key, value=None, target_location=None, target_sizes=None):
        raise NotImplementedError("KVCR requires indexed HiCache host pages")

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise NotImplementedError("KVCR requires indexed HiCache host pages")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_poll.set()
        # Do not hold the IO lock while joining its completion-poll owner.
        if self._poll_thread is not None:
            self._poll_thread.join()
        with self._lock:
            if self._runner is not None:
                try:
                    self.release_all_lookup_claims()
                    if self._diagnostic_telemetry:
                        self._drain_diagnostics(force=True)
                    self._runner.close()
                except Exception:
                    logger.critical(
                        "KVCR L3 close failed; retaining buffers until process exit",
                        exc_info=True,
                    )
                    os._exit(1)
            for allocation in self._maps:
                allocation.close()
            self._maps.clear()
