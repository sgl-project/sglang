# SPDX-License-Identifier: Apache-2.0
"""Host-staged KVCR L3: SGLang HBM -> HiCache host -> KVCR DRAM.

Reads deliver directly into the host slots owned/locked by HiCache. KVCR
serves only its own deposited DRAM, never unpinned framework host pages.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import math
import mmap
import os
import socket
import threading
import uuid
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from queue import Empty, SimpleQueue

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.storage.kvcr.endpoints import (
    control_address,
    rank_endpoint,
    rank_offset,
    require_routable_host,
)
from sglang.srt.mem_cache.storage.kvcr.hints import (
    fetch_hint,
    parse_router_hint,
    peer_hint,
    router_hash,
)
from sglang.srt.mem_cache.storage.kvcr.host_layout import (
    StorageKeyAdapter,
    block_key,
    describe_pool,
    page_slots,
    resume_boundaries,
    storage_namespace,
)
from sglang.srt.mem_cache.storage.kvcr.inventory import KVCRInventory
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# Waiters poll at this interval themselves. The background poll runs at it only
# after draining completions and otherwise backs off to the cap: each wakeup
# takes the GIL from the scheduler. A peer asking for pages this rank lacks
# waits up to the source cap for its refusal; without a control endpoint there
# are no peers, only lease releases.
_POLL_INTERVAL_SECONDS = 0.001
_IDLE_POLL_MAX_SECONDS = 0.02
_IDLE_POLL_MAX_SECONDS_WITHOUT_PEERS = 0.1

# Parsed router hints kept per (envelope, mapping) object pair: the prefetch
# thread's operation and the scheduler's lookup may interleave. Arbitrary.
_ROUTER_HINT_CACHE_SIZE = 8

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

# Pools HiCache backs up on every rank even where it elects one writer for
# rank-replicated KV (HybridCacheController.should_backup): their bytes are
# TP-sharded, so each rank keeps its own. So are MHA draft pools, by type.
_RANK_SHARDED_POOLS = frozenset({str(PoolName.MAMBA)})
_MHA_DRAFT_POOLS = frozenset({str(PoolName.DRAFT), str(PoolName.DRAFT_SWA)})

# Default KVCR deadlines scale with one HiCache storage batch (128 pages of
# every pool): the operation deadline is that batch at this per-rank rate, and
# a transfer still in flight at abandonment ends the worker (see README).
_SLOW_TRANSFER_BYTES_PER_SECOND = 250 * 1000**2
_MIN_OPERATION_TIMEOUT_MS = 5000
_MIN_ABANDON_TIMEOUT_MS = 30000
_ABANDON_PER_OPERATION = 3

_KVCR_REQUIRED = (
    "KVCR HiCache backend requires the kvcr>=0.2.0 package and its NIXL Python "
    "bindings (0.1.x lacks the API this backend uses)"
)

# Extra-config keys this backend reads.
_EXTRA_CONFIG_KEYS = frozenset(
    {
        "abandon_timeout_ms",
        "control_advertise_host",
        "control_host",
        "control_port",
        "enable_remote_hint",
        "extra_backend_tag",
        "local_dram_bytes",
        "local_dram_pool_pages",
        "nixl_listen_port",
        "operation_timeout_ms",
        "publish_inventory",
        "share_replicated_kv",
        "startup_timeout_seconds",
    }
)
# Generic HiCache keys; decode offload passes them through unpopped.
_HICACHE_GENERIC_KEYS = frozenset(
    {
        "prefetch_threshold",
        "prefetch_timeout_base",
        "prefetch_timeout_per_ki_token",
        "hicache_storage_pass_prefix_keys",
        # Read by the controller when KVCR is loaded as a "dynamic" backend.
        "interface_v1",
        "tp_lcm_size",
        "file_storage_path",
    }
)


def _bool_option(options: dict, name: str, default: bool = False) -> bool:
    value = options.get(name, default)
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

    # Objects of rank-sharded pools are keyed by this rank's shard, so every
    # rank backs up its own copy. Rank-replicated pools (MLA) are shared
    # unless share_replicated_kv is off; see _share_replicated.
    requires_rank_local_backup = True
    rank_keyed_mha_drafts = True

    def __init__(self, storage_config, mem_pool_host):
        if storage_config.should_split_heads:
            raise ValueError("KVCR does not support heterogeneous TP head splitting")
        if storage_config.dcp_size != 1:
            raise ValueError("KVCR HiCacheStorage does not support DCP yet")
        if storage_config.pp_size != 1:
            raise ValueError("KVCR HiCacheStorage currently requires PP=1")
        if is_dp_attention_enabled() and get_parallel().dp_size > 1:
            # Every replica would number its ranks by attention DP alone and
            # bind the same control ports.
            raise ValueError(
                "KVCR HiCacheStorage does not support replica DP (--dp-size) "
                "combined with attention DP"
            )
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
        self._pools_attached = False
        self._runner = None
        self._maps = []
        # Serializes every KVCR call: its API allows one caller at a time, but it
        # is not thread-affine (NIXL and ZMQ sockets live on KVCR's progress thread).
        self._lock = threading.RLock()
        self._completion = threading.Condition(self._lock)
        self._completed = {}
        self._poll_thread = None
        self._stop_poll = threading.Event()
        self._closed = False
        self._pinning = _NoFrameworkPinning()
        self._extra = dict(storage_config.extra_config or {})
        unknown = sorted(set(self._extra) - _EXTRA_CONFIG_KEYS - _HICACHE_GENERIC_KEYS)
        if unknown:
            logger.warning("KVCR ignores unknown extra-config keys: %s", unknown)
        self._remote_hint = _bool_option(self._extra, "enable_remote_hint")
        if self._remote_hint and get_parallel().nnodes > 1:
            # Peers reach shard k at shard 0's advertised host, port + k.
            raise ValueError(
                "KVCR enable_remote_hint (and publish_inventory) requires --nnodes 1"
            )
        self._shared = self._share_replicated(storage_config)
        # The controller elects the writer from the config it holds; with CP=1
        # (required to share) KVCR's own config does not renumber it.
        self._writer = storage_config.is_storage_writer
        if self._shared and self._writer != (rank_offset(storage_config) == 0):
            raise ValueError("KVCR expects the replica's first shard to write")
        self.requires_rank_local_backup = not self._shared
        if "startup_timeout_seconds" in self._extra and (
            type(self._extra["startup_timeout_seconds"]) is not int
            or self._extra["startup_timeout_seconds"] <= 0
        ):
            raise ValueError("startup_timeout_seconds must be a positive integer")
        tag = self._extra.get("extra_backend_tag")
        if tag is not None and (type(tag) is not str or not tag):
            raise ValueError("KVCR extra_backend_tag must be a non-empty string")
        self._offset = rank_offset(storage_config)
        # Fail the attach, not every later storage call, on a static
        # misconfiguration. Memory registration stays at the first IO: host
        # pools (e.g. a draft sidecar) may still register.
        try:
            # Exported from 0.2.0 on; the import also loads its NIXL bindings.
            from kvcr import ZmqPeerControlChannel
        except ImportError as e:
            raise ImportError(_KVCR_REQUIRED) from e
        if not hasattr(ZmqPeerControlChannel, "adopt_listener"):
            # Takes the listener reserved at attach (_reserve_ports); 0.2.0 on.
            raise ImportError(_KVCR_REQUIRED)
        control_host = self._extra.get("control_host")
        if control_host is not None and (
            not isinstance(control_host, str) or ":" in control_host
        ):
            # KVCR 0.2.0's control channel binds IPv4 (tcp://host:port).
            raise ValueError("KVCR control_host must be an IPv4 address or host name")
        self._budget = self._extra.get("local_dram_bytes", 8 << 30)
        self._pages_by_pool = self._extra.get("local_dram_pool_pages", {})
        if (
            type(self._budget) is not int
            or self._budget <= 0
            or not isinstance(self._pages_by_pool, dict)
            or any(type(v) is not int or v < 1 for v in self._pages_by_pool.values())
        ):
            raise ValueError("Invalid KVCR local DRAM budget")
        # Unset deadlines are derived from the pools once they are known
        # (_timeouts); set ones must already satisfy KVCR's own rule here, so
        # that it cannot fail the first IO.
        self._operation_timeout_ms = self._extra.get("operation_timeout_ms")
        self._abandon_timeout_ms = self._extra.get("abandon_timeout_ms")
        for name, value in (
            ("operation_timeout_ms", self._operation_timeout_ms),
            ("abandon_timeout_ms", self._abandon_timeout_ms),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"KVCR {name} must be a positive integer")
        if self._abandon_timeout_ms is not None and self._abandon_timeout_ms < 2 * (
            self._operation_timeout_ms or 1
        ):
            raise ValueError(
                "KVCR abandon_timeout_ms must be at least twice operation_timeout_ms"
            )
        self._control = control_address(self.config, self._extra)
        port = self._control[1]
        dp_stride = self.config.tp_size * self.config.attn_cp_size * self.config.pp_size
        offset = self.config.dp_rank * dp_stride + self._offset
        nixl_base_port = self._extra.get("nixl_listen_port", port - offset + 1024)
        if type(nixl_base_port) is not int or not 0 < nixl_base_port <= 65535:
            raise ValueError("KVCR nixl_listen_port must be a valid integer port")
        self._nixl_port = nixl_base_port + offset
        if not 0 < self._nixl_port <= 65535 or port == self._nixl_port:
            raise ValueError("KVCR control/NIXL port range is invalid or overlaps")
        self._agent_name = f"sglang-kvcr-{uuid.uuid4().hex}"
        # A shared rank's source: the replica's first shard, on this host.
        self._writer_endpoint = None
        if self._shared and not self._writer:
            self._writer_endpoint = rank_endpoint(
                f"tcp://{self._peer_host()}:{port}", -self._offset
            )
            if self._writer_endpoint is None:
                raise ValueError("KVCR cannot address the replica's first shard")
        self._idle_poll_max = (
            _IDLE_POLL_MAX_SECONDS
            if self._needs_control()
            else _IDLE_POLL_MAX_SECONDS_WITHOUT_PEERS
        )
        # Publish this shard's DRAM residency as KV events (ownership "kvcr")
        # for an external KV state agent, e.g. Dynamo's.
        self.publishes_kv_events = _bool_option(self._extra, "publish_inventory")
        if self.publishes_kv_events:
            if not self._remote_hint or "control_advertise_host" not in self._extra:
                raise ValueError(
                    "KVCR publish_inventory requires enable_remote_hint and an "
                    "explicit control_advertise_host"
                )
            # The state agent subscribes to the events on this host.
            require_routable_host(self._control[0])
        self._inventory = KVCRInventory() if self.publishes_kv_events else None
        # Each attention TP x CP rank stores its own shard (no PP, no DCP), and a
        # page is advertised once every shard holds it. Shared replicated pools
        # live on the first shard alone; the others are its replicas.
        self.kv_event_shard = self._offset
        self.kv_event_shard_count = (
            1 if self._shared else self.config.tp_size * self.config.attn_cp_size
        )
        # External hints name the source replica's first shard; a rank reads
        # its shared pools from that replica's first shard too.
        self._source_offset = 0 if self._shared else self._offset
        self._router_hints = OrderedDict()
        self._rank_local_pools = set(_RANK_SHARDED_POOLS)
        # This shard's and its writer's storage namespaces, fixed at start.
        self._namespace = self._shared_namespace = None
        # The control listener and the NIXL port, held from the attach until
        # KVCR binds them at start (_reserve_ports).
        self._listener = self._nixl_hold = None
        # Set once advertised; fixed for this backend's (and engine's) lifetime.
        self._cache_owner_id = None
        self.register_mem_pool_host(mem_pool_host)

    def _share_replicated(self, storage_config) -> bool:
        """Keep one copy of rank-replicated pools per replica, on its writer.

        Like HiCache's other backends, which elect one writer for MLA KV, but
        a KVCR rank holds only its own DRAM: the other ranks restore those
        pages from the writer's KVCR on the same host, through its control
        endpoint. Pools HiCache keeps per rank stay per rank. Opt-in: it opens
        that endpoint and makes the writer every sibling's source (README).
        """
        copies = storage_config.tp_size * storage_config.attn_cp_size
        if not storage_config.is_rank_replicated or copies == 1:
            return False
        reason = None
        if not _bool_option(self._extra, "share_replicated_kv"):
            reason = "share_replicated_kv is off"
        elif storage_config.attn_cp_size != 1:
            reason = "attention CP > 1"
        elif get_parallel().nnodes != 1:
            # Ranks reach the writer at its port on their own host.
            reason = "--nnodes > 1"
        if reason is not None:
            logger.warning(
                "KVCR keeps a copy of rank-replicated KV on every attention rank "
                "(%d copies per page in L3 DRAM and backup traffic) because %s",
                copies,
                reason,
            )
            return False
        return True

    def _shares(self, pool: str) -> bool:
        return self._shared and pool not in self._rank_local_pools

    def _namespaces(self, layouts=None) -> tuple[str, str]:
        """(this shard's, the writer's) storage namespace; equal on the writer."""
        layouts = self._layouts if layouts is None else layouts
        own = storage_namespace(self.config, layouts, shared=self._shared)
        if not self._shared or self._writer:
            return own, own
        writer = dataclasses.replace(self.config, tp_rank=0)
        return own, storage_namespace(writer, layouts, shared=True)

    def _key(self, page_key: str, pool: str) -> bytes:
        namespace = self._shared_namespace if self._shares(pool) else self._namespace
        return block_key(page_key, namespace, pool)

    def consumes_kv_hints(self) -> bool:
        return self._remote_hint

    def get_kv_hint_source(self):
        """This replica's router source, reported by its first shard only.

        Peers reach the other shards by rank offset from this endpoint. Only
        with remote hints on and an explicit control_advertise_host: the
        default host is loopback.
        """
        if (
            self._offset != 0
            or not self._remote_hint
            or "control_advertise_host" not in self._extra
        ):
            return None
        host, port = self._control
        source = {
            "dp_rank": int(self.config.dp_rank),
            "capabilities": ["router_hint"],
            "source_control_endpoint": f"tcp://{host}:{port}",
        }
        if self.publishes_kv_events:
            source["capabilities"].append("kvcr_inventory")
            source["cache_owner_id"] = self._owner_id()
        return source

    def _owner_id(self) -> str:
        """Dynamo CacheOwnerId: <ns[:32]>:<ns[32:]>/<16 zeros>/<incarnation>.

        The indexer domain is the 64-hex storage namespace, so engines that
        can serve each other's pages share it; the cluster field is fixed.
        KVCR's DRAM is process-local: the incarnation is this process's agent
        UUID, so a new engine never inherits retained residency.
        """
        with self._lock:
            if self._cache_owner_id is None:
                if not self._layouts:
                    raise ValueError("KVCR has no registered physical host pools")
                namespace = self._namespaces()[1]
                self._cache_owner_id = (
                    f"{namespace[:32]}:{namespace[32:]}/{'0' * 16}/"
                    f"{self._agent_name.rsplit('-', 1)[-1]}"
                )
            return self._cache_owner_id

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
            if self._cache_owner_id is not None:
                # The advertised owner names the namespace of these pools.
                raise ValueError("KVCR host pools must be registered before discovery")
            layout = describe_pool(name, host_pool)
            if name in _MHA_DRAFT_POOLS and isinstance(host_pool, MHATokenToKVPoolHost):
                # Its heads are this rank's (should_backup writes it everywhere).
                if self._shared and self.publishes_kv_events:
                    # Only the writer's residency is published, which says
                    # nothing of the other ranks' own draft pages.
                    raise ValueError(
                        "KVCR share_replicated_kv with publish_inventory does "
                        f"not support an MHA {name} pool yet"
                    )
                self._rank_local_pools.add(name)
            if self._pools_attached:
                # A late sidecar shrinks every pool's share: fail its
                # registration, not the first IO.
                self._pool_pages({**self._layouts, name: layout})
            self._layouts[name] = layout
            self.registered_pools[name] = host_pool

    def on_host_pools_registered(self) -> None:
        # Fail the attach if the budget cannot give a registered pool one page.
        if self._layouts:
            self._pool_pages()
        self._reserve_ports()
        self._pools_attached = True

    def _needs_control(self) -> bool:
        # KVCR serves and fetches peer pages only through its control endpoint.
        return self._remote_hint or self._shared

    def _reserve_ports(self) -> None:
        """Fail the attach, not the first IO, when a listen port is taken.

        KVCR starts (and binds) at the first IO, after the worker reported its
        sources and passed health checks. The control listener is bound here
        and handed to KVCR at start. The NIXL port stays bound, not listening,
        until KVCR binds it, so no outgoing connection takes it meanwhile
        (by default it lies in Linux's ephemeral port range).
        """
        if self._nixl_hold is not None or self._runner is not None:
            return
        host, port = self._control
        try:
            with ExitStack() as reserved:
                hold = reserved.enter_context(socket.socket())
                # As listeners bind it: only a live socket on the port collides.
                hold.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                hold.bind(("", self._nixl_port))
                if self._needs_control():
                    self._listener = reserved.enter_context(
                        socket.create_server((self._control_bind_host(), port))
                    )
                reserved.pop_all()
        except OSError:
            # One rank failing a runtime attach would leave the others in its
            # collectives; at launch this fails the launch all the same.
            logger.critical(
                "KVCR cannot bind its control port %d or NIXL port %d; "
                "terminating worker",
                port,
                self._nixl_port,
                exc_info=True,
            )
            os._exit(1)
        self._nixl_hold = hold

    def _control_bind_host(self) -> str:
        # Only the replica's own ranks dial a shared-only endpoint.
        default = "0.0.0.0" if self._remote_hint else "127.0.0.1"
        return self._extra.get("control_host", default)

    def _peer_host(self) -> str:
        """The host peers dial and sources reply to.

        The advertised host with remote hints; otherwise only this replica's
        ranks dial it, at the host it is bound to.
        """
        if self._remote_hint:
            return self._control[0]
        host = self._control_bind_host()
        return "127.0.0.1" if host in ("0.0.0.0", "") else host

    def _timeouts(self, layouts=None) -> tuple[int, int]:
        """KVCR's operation and abandonment deadlines, in ms.

        Unset ones scale with one storage batch, so large pages get more time
        before a slow but healthy transfer counts as uncertain.
        """
        layouts = self._layouts if layouts is None else layouts
        # Per rank, also with shared replicated KV: scaling by the writer's
        # fan-in would stretch a dead peer's stall by TP; the rate is already
        # far below measured same-host and peer reads.
        batch_bytes = STORAGE_BATCH_SIZE * sum(
            component.size
            for layout in layouts.values()
            for component in layout.components
        )
        derived = max(
            _MIN_OPERATION_TIMEOUT_MS,
            math.ceil(batch_bytes * 1000 / _SLOW_TRANSFER_BYTES_PER_SECOND),
        )
        operation, abandon = self._operation_timeout_ms, self._abandon_timeout_ms
        if operation is None:
            operation = derived if abandon is None else min(derived, abandon // 2)
        if abandon is None:
            abandon = max(_MIN_ABANDON_TIMEOUT_MS, _ABANDON_PER_OPERATION * operation)
        return operation, abandon

    def _warn_unmatched_pool_pages(self) -> None:
        """Report local_dram_pool_pages names once registration has closed.

        Draft sidecar pools register after the attach, so this runs at start.
        """
        unmatched = sorted(set(self._pages_by_pool) - set(self._layouts))
        if unmatched:
            logger.warning(
                "KVCR local_dram_pool_pages names no registered pool: %s "
                "(registered: %s)",
                unmatched,
                sorted(self._layouts),
            )

    def _pool_pages(self, layouts=None) -> dict:
        """KVCR DRAM pages per physical pool (the registered ones by default)."""
        layouts = self._layouts if layouts is None else layouts
        total = sum(
            component.size * component.count
            for layout in layouts.values()
            for component in layout.components
        )
        pages_by_pool = {}
        for name, layout in layouts.items():
            pages = self._pages_by_pool.get(
                name, self._budget * layout.components[0].count // total
            )
            if pages < 1:
                raise ValueError(f"KVCR DRAM budget cannot hold one page of {name}")
            pages_by_pool[name] = pages
        return pages_by_pool

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
        self._warn_unmatched_pool_pages()
        operation_timeout_ms, abandon_timeout_ms = self._timeouts()
        self._namespace, self._shared_namespace = self._namespaces()
        if self._inventory is not None:
            # Only the writer's pages are advertised (kv_event_shard_count).
            self._inventory.configure(
                self._shared_namespace,
                [name for name in self._layouts if name not in _TRAILING_POOLS],
            )
        port = self._control[1]
        layouts, regions, plans = [], [], []
        pool_pages = self._pool_pages()
        # KVCR's DRAM pools must match its pool layouts, which name every
        # pool a restore lands in. A replica keeps one page of a shared pool
        # beside its own pools, and no DRAM if it has none.
        replica = [
            name for name in pool_pages if not self._writer and self._shares(name)
        ]
        if len(replica) == len(pool_pages):
            pool_pages = {name: 0 for name in pool_pages}
        else:
            pool_pages.update({name: 1 for name in replica})
        for name, pages in pool_pages.items():
            layout = self._layouts[name]
            for component in layout.components:
                if pages:
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
        # Peers fetch through the control endpoint, unauthenticated, so it
        # exists only with remote hints or shared replicated pools; local DRAM
        # IO does not use it. A rank without DRAM needs it to start at all.
        channel = None
        if self._needs_control():
            channel = ZmqPeerControlChannel(
                bind_host=self._control_bind_host(),
                bind_port=port,
                # Sources reply here.
                advertise_host=self._peer_host(),
            )
        allocations, mappings = [], []
        with ExitStack() as resources:
            if channel is not None:
                resources.callback(channel.close)
            for name, length in plans:
                allocation = _allocate_dram_pool(length)
                resources.callback(allocation.close)
                mappings.append(allocation)
                address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
                allocations.append((name, address, length))
            if self._nixl_hold is not None:
                self._nixl_hold.close()
                self._nixl_hold = None
            try:
                if channel is not None and self._listener is not None:
                    listener, self._listener = self._listener, None
                    channel.adopt_listener(listener.detach())
                self._runner = KVCR(
                    config=KVCRConfig(
                        nixl_agent_name=self._agent_name,
                        nixl_listen_port=self._nixl_port,
                        pool_layouts=layouts,
                        operation_timeout_ms=operation_timeout_ms,
                        abandon_timeout_ms=abandon_timeout_ms,
                        startup_timeout_seconds=self._extra.get(
                            "startup_timeout_seconds"
                        ),
                    ),
                    bindings=KVCRBindings(
                        request_pin=self._pinning.request_pin,
                        poll_pin_results=self._pinning.poll_pin_results,
                        release_pin=self._pinning.release_pin,
                        framework_control=channel,
                        key_adapter=StorageKeyAdapter(),
                        on_resilience_event=self._resilience_event,
                        inventory_sink=(
                            self._record_inventory
                            if self._inventory is not None
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
            "KVCR L3 ready: endpoint=%s physical_pools=%s bytes=%d namespace=%s "
            "shared_replicated=%s writer=%s operation_timeout_ms=%d "
            "abandon_timeout_ms=%d",
            channel.endpoint if channel is not None else None,
            layouts,
            sum(size for _, _, size in allocations),
            self._namespace,
            self._shared,
            self._writer_endpoint or "self",
            operation_timeout_ms,
            abandon_timeout_ms,
        )

    @staticmethod
    def _resilience_event(error):
        from kvcr.types import TransferError

        if isinstance(error, TransferError) and error.state == "uncertain":
            # HiCache cannot quarantine arbitrary source/destination pages yet.
            # Returning a miss would let it reuse memory still exposed to NIXL.
            logger.critical("Unsafe KVCR transfer; terminating worker: %s", error)
            os._exit(1)
        if "source writes disabled" in str(error):
            # KVCR 0.2.0 latches a stalled source off until restart: every
            # peer, including this replica's other ranks, then misses here.
            # 0.2.0 reports it only as this RuntimeError message (from
            # dangling_ops.check_source_progress); re-check it on a KVCR bump.
            logger.error("KVCR stopped serving peers until restart: %s", error)
            return
        logger.warning("KVCR resilience event: %s", error)

    def _router_hint(self, extra_info, page_keys, offset=None):
        """The request's router hint for a source shard, narrowed to these pages.

        The shard is the source replica's first (for shared pools) or this
        rank's own offset there. The controller rebuilds extra_info for every
        storage batch but carries the operation's envelope object, so it is
        parsed once per operation and offset; the cache holds it, so its id is
        not reused while cached. Its key mapping is read per batch: a lookup
        builds a new one each time. Called under self._lock.
        """
        extra = extra_info.extra_info if extra_info is not None else None
        if not self._remote_hint or not extra or extra.get("kv_hints") is None:
            return None
        envelope = extra["kv_hints"]
        offset = self._source_offset if offset is None else offset
        key = (id(envelope), offset)
        cached = self._router_hints.get(key)
        if cached is None or cached[0] is not envelope:
            cached = (envelope, parse_router_hint(envelope, offset))
            self._router_hints[key] = cached
            if len(self._router_hints) > _ROUTER_HINT_CACHE_SIZE:
                self._router_hints.popitem(last=False)
        else:
            self._router_hints.move_to_end(key)
        return peer_hint(cached[1], page_keys, extra.get("storage_key_to_event_hash"))

    def _writer_hint(self, page_keys):
        """Fetch these pages from this replica's writer."""
        return fetch_hint(
            self._writer_endpoint, dict.fromkeys(map(router_hash, page_keys))
        )

    @contextmanager
    def _hint(self, hint):
        request_id = None
        if hint is not None:
            request_id = uuid.uuid4().hex
            self._runner.submit_hint(hint, request_id=request_id)
        try:
            yield request_id
        finally:
            if request_id is not None:
                self._runner.discard_hint(request_id)

    def _deliver(self, blocks, pages, shared, extra_info):
        """One terminal read of these blocks (identity -> page key in pages).

        KVCR binds one source per request. A replica rank reads its shared
        pools as the writer would, local first: from this replica's writer,
        then pages it lacks that a router hint names from the hinted replica's
        writer. Its own pools come from local DRAM or, hinted, from its shard
        of the hinted replica, in flight beside the shared read.
        """
        if self._writer_endpoint is None:
            hint = self._router_hint(extra_info, list(dict.fromkeys(pages.values())))
            with self._hint(hint) as request_id:
                return self._wait(self._runner.deliver(blocks, request_id=request_id))
        own = {key: refs for key, refs in blocks.items() if key not in shared}
        local = {key: refs for key, refs in blocks.items() if key in shared}
        own_hint = self._router_hint(
            extra_info, [pages[key] for key in own], offset=self._offset
        )
        result = self._deliver_parts(
            ((own, own_hint), (local, self._replica_hint(local, pages)))
        )
        missed = {
            key: refs
            for key, refs in local.items()
            if key not in result or not result[key].success
        }
        remote = self._router_hint(extra_info, [pages[key] for key in missed])
        if missed and remote is not None:
            result.update(self._deliver_parts(((missed, remote),)))
        return result

    def _replica_hint(self, blocks, pages):
        return self._writer_hint([pages[key] for key in blocks]) if blocks else None

    def _deliver_parts(self, parts):
        """Deliver each (blocks, hint) part under its own request; wait for all."""
        with ExitStack() as scopes:
            handles = []
            for blocks, hint in parts:
                if blocks:
                    request_id = scopes.enter_context(self._hint(hint))
                    handles.append(self._runner.deliver(blocks, request_id=request_id))
            result = {}
            for handle in handles:
                result.update(self._wait(handle))
        return result

    def _wait(self, handle):
        # There can be concurrent backup and prefetch threads. One caller may
        # drain another caller's completion; keep it until the owner consumes it.
        with self._completion:
            while handle not in self._completed:
                self._completed.update(self._runner.poll_completed())
                if handle not in self._completed:
                    self._completion.wait(timeout=_POLL_INTERVAL_SECONDS)
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
                    busy = bool(completed)
            except Exception:
                # Native progress failure cannot safely be treated as a cache
                # miss: HiCache could recycle a buffer still exposed to NIXL.
                logger.critical(
                    "KVCR L3 progress failed; terminating worker", exc_info=True
                )
                os._exit(1)
            wait = (
                _POLL_INTERVAL_SECONDS if busy else min(wait * 2, self._idle_poll_max)
            )
            self._stop_poll.wait(wait)

    def _record_inventory(self, event):
        # KVCR calls its sink from deposit/deliver/poll_completed, which run
        # under self._lock. Only its local DRAM tier ("DRAM") is advertised.
        if event.tier.value == "DRAM":
            self._inventory.update(event)

    def pending_kv_events(self):
        # Called every scheduler step. A racy length is fine: events appended
        # after this read are drained at the next step.
        return len(self._inventory.events) if self._inventory is not None else 0

    def take_kv_events(self):
        # Lock-free: storage threads may hold self._lock across native calls,
        # and this runs on the scheduler thread before a collective gather.
        return self._inventory.take() if self._inventory is not None else []

    @contextmanager
    def _inventory_operation(self, extra_info, blocks):
        """Hold the IO's page identities while it may change residency."""
        if self._inventory is None:
            yield
            return
        extra = getattr(extra_info, "extra_info", None) or {}
        retained = self._inventory.register(
            extra.get("kv_event_metadata") or {}, blocks
        )
        try:
            yield
        finally:
            self._inventory.release(retained)

    def _io(self, transfers, *, is_set, extra_info):
        from kvcr.types import MemoryRef

        with self._lock:
            self._start()
            blocks, groups, pages, shared = {}, {}, {}, set()
            for transfer in transfers:
                name = str(transfer.name)
                keys = transfer.keys or []
                if name == str(PoolName.KV) and name not in self._layouts:
                    groups[name] = [(None, True) for _ in keys]
                    continue
                if is_set and keys and self._shares(name) and not self._writer:
                    # HiCache elects one writer (backup_skip); this is a bug.
                    raise ValueError(f"Only the writer shard backs up shared {name}")
                layout = self._layouts[name]
                slots = page_slots(layout, keys, transfer.host_indices)
                groups.setdefault(name, [])
                for key, slot in zip(keys, slots, strict=True):
                    identity = self._key(key, name)
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
                    pages[identity] = key
                    if self._shares(name):
                        shared.add(identity)
                    groups[name].append((identity, False))
            if not blocks:
                return {name: [True] * len(group) for name, group in groups.items()}
            # _wait releases the condition's recursive lock while sleeping.
            with self._inventory_operation(extra_info, blocks):
                if is_set:
                    # Writes carry no kv_hints; deposits are always local.
                    result = self._wait(self._runner.deposit(blocks))
                else:
                    result = self._deliver(blocks, pages, shared, extra_info)
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
        # Shared pools are the writer's to answer: every rank's lookup is
        # MIN-reduced (or, for trailing pools, intersected) across the
        # replica's attention ranks, so this rank claims every page and the
        # writer's own answer decides.
        replica = self._shared and not self._writer
        queried = [t for t in required if not (replica and self._shares(str(t.name)))]
        statuses = {}
        with self._lock:
            # A replica starts here too, not in its first (critical-path) read.
            self._start()
            if queried:
                identities = [
                    self._key(key, str(transfer.name))
                    for transfer in queried
                    for key in keys
                ]
                # A replica queries only its own pools: at its own shard.
                hint = self._router_hint(
                    extra_info, keys, offset=self._offset if replica else None
                )
                with self._hint(hint) as request_id:
                    flat = self._runner.query(identities, request_id=request_id)
            for index, transfer in enumerate(queried):
                statuses[id(transfer)] = flat[
                    index * len(keys) : (index + 1) * len(keys)
                ]
        restorable, hits = set(range(1, len(keys) + 1)), {}
        for transfer in required:
            status = statuses.get(id(transfer))
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
            exists = (
                [True] * len(keys)
                if status is None
                else [s in accepted for s, _ in status]
            )
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
            # Reserved but never handed to KVCR.
            for reserved in (self._listener, self._nixl_hold):
                if reserved is not None:
                    reserved.close()
            self._listener = self._nixl_hold = None
