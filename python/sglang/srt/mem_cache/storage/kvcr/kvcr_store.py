# SPDX-License-Identifier: Apache-2.0
"""Host-staged KVCR L3: SGLang HBM -> HiCache host -> KVCR DRAM.

Reads deliver directly into the host slots owned/locked by HiCache. KVCR
serves only its own deposited DRAM, never unpinned framework host pages.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import mmap
import os
import threading
import uuid
from contextlib import ExitStack, contextmanager
from queue import Empty, SimpleQueue

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.managers.kv_hints import decode_kv_hints_envelope
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.storage.kvcr.endpoints import control_address, rank_offset
from sglang.srt.mem_cache.storage.kvcr.hints import peer_hint
from sglang.srt.mem_cache.storage.kvcr.host_layout import (
    StorageKeyAdapter,
    block_key,
    describe_pool,
    page_slots,
    resume_boundaries,
    storage_namespace,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# The completion poll runs at this interval while callers wait or completions arrive,
# and backs off to the cap while idle: each wakeup takes the GIL from the scheduler.
_POLL_INTERVAL_SECONDS = 0.001
_IDLE_POLL_MAX_SECONDS = 0.02

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

    # Objects are keyed by this rank's shard, so every rank backs up its own
    # copy, including rank-replicated MLA KV.
    requires_rank_local_backup = True

    def __init__(self, storage_config, mem_pool_host):
        if storage_config.should_split_heads:
            raise ValueError("KVCR does not support heterogeneous TP head splitting")
        if storage_config.dcp_size != 1:
            raise ValueError("KVCR HiCacheStorage does not support DCP yet")
        if storage_config.pp_size != 1:
            raise ValueError("KVCR HiCacheStorage currently requires PP=1")
        if not is_dp_attention_enabled():
            # Without attention DP, HiCache leaves the replica index at 0 and
            # numbers TP across CP. Rank endpoints need both separately.
            parallel = get_parallel()
            changes = {"dp_rank": parallel.dp_rank or 0}
            if storage_config.attn_cp_size > 1:
                changes.update(
                    tp_rank=parallel.attn_tp_rank, tp_size=parallel.attn_tp_size
                )
            storage_config = dataclasses.replace(storage_config, **changes)
        self.config = storage_config
        self.mem_pool_host = mem_pool_host
        self.registered_pools = {}
        self._layouts = {}
        self._runner = None
        self._maps = []
        self._lock = threading.RLock()
        self._completion = threading.Condition(self._lock)
        self._completed = {}
        self._poll_thread = None
        self._stop_poll = threading.Event()
        # Callers blocked in _wait; the poll loop stays at its short interval while any wait.
        self._waiters = 0
        self._closed = False
        self._pinning = _NoFrameworkPinning()
        self._extra = dict(storage_config.extra_config or {})
        self._remote_hint = _bool_option(self._extra, "enable_remote_hint")
        if "startup_timeout_seconds" in self._extra and (
            type(self._extra["startup_timeout_seconds"]) is not int
            or self._extra["startup_timeout_seconds"] <= 0
        ):
            raise ValueError("startup_timeout_seconds must be a positive integer")
        self._offset = rank_offset(storage_config)
        self._agent_name = f"sglang-kvcr-{uuid.uuid4().hex}"
        self.register_mem_pool_host(mem_pool_host)

    def consumes_kv_hints(self) -> bool:
        return self._remote_hint

    def register_mem_pool_host(self, mem_pool_host):
        self.mem_pool_host = mem_pool_host
        # Hybrid logical anchors have no payload. The real pools arrive via v2.
        if mem_pool_host.kv_buffer is not None:
            self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        name = str(host_pool_name)
        with self._lock:
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

        if not self._layouts:
            raise ValueError("KVCR has no registered physical host pools")
        self._namespace = storage_namespace(self.config, self._layouts)
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
                plans.append((component.name, pages * component.size))
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

    @contextmanager
    def _hint(self, extra_info, keys=None):
        request_id = None
        hint = peer_hint(extra_info, self._offset, keys) if self._remote_hint else None
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
            self._waiters += 1
            try:
                while handle not in self._completed:
                    self._completed.update(self._runner.poll_completed())
                    if handle not in self._completed:
                        self._completion.wait(timeout=_POLL_INTERVAL_SECONDS)
            finally:
                self._waiters -= 1
            result = self._completed.pop(handle)
            self._completion.notify_all()
            return result

    def _poll_loop(self):
        wait = _POLL_INTERVAL_SECONDS
        while not self._stop_poll.is_set():
            busy = True
            try:
                with self._completion:
                    completed = list(self._runner.poll_completed())
                    self._completed.update(completed)
                    self._completion.notify_all()
                    busy = bool(completed) or self._waiters > 0
            except Exception:
                # Native progress failure cannot safely be treated as a cache
                # miss: HiCache could recycle a buffer still exposed to NIXL.
                logger.critical(
                    "KVCR L3 progress failed; terminating worker", exc_info=True
                )
                os._exit(1)
            wait = (
                _POLL_INTERVAL_SECONDS
                if busy
                else min(wait * 2, _IDLE_POLL_MAX_SECONDS)
            )
            self._stop_poll.wait(wait)

    def _io(self, transfers, *, is_set, extra_info):
        from kvcr.types import MemoryRef

        with self._lock:
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
            # _wait releases the condition's recursive lock while sleeping.
            if is_set:
                # Writes carry no kv_hints; deposits are always local.
                result = self._wait(self._runner.deposit(blocks))
            else:
                with self._hint(extra_info, blocks) as request_id:
                    handle = self._runner.deliver(blocks, request_id=request_id)
                    result = self._wait(handle)
        return {
            name: [
                logical or (key in result and result[key].success)
                for key, logical in group
            ]
            for name, group in groups.items()
        }

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
            trailing = None
            # FETCHABLE only means a router hint named the logical page; it
            # does not prove the peer holds this pool's checkpoint. A resume
            # boundary needs that state, so trailing pools count local hits.
            accepted = (QueryStatus.HIT, QueryStatus.FETCHABLE)
            if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys or []))
                accepted = (QueryStatus.HIT,)
            elif transfer.hit_policy != PoolHitPolicy.ALL_PAGES:
                raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")
            exists = [s in accepted for s, _ in status]
            valid = resume_boundaries(exists, trailing=trailing)
            hits[str(transfer.name)] = max(valid, default=0)
            restorable.intersection_update(valid)
        valid = sorted(restorable)
        return PoolTransferResult(max(valid, default=0), hits, valid)

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
