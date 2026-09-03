# SPDX-License-Identifier: Apache-2.0
"""KVCR as a SGLang HiCacheStorage backend (DRAFT / WIP).

KVCR is an asymmetric peer-to-peer G2 KV coordinator (the ``nvidia-kvcr`` wheel,
package ``kvcr``).
This module adapts it to SGLang's content-addressed ``HiCacheStorage`` contract.

Mapping (see the module docstring sections below for the honest status of each):

    HiCacheStorage                     KVCR core
    ------------------------------     --------------------------------------
    batch_set_v2  (host -> storage)    deposit()  -> KVCR-owned local DRAM slots
    batch_get_v2  local hit            deliver() -> local get-back (slot->host)
    batch_get_v2  remote hit           submit_hint() + deliver() (NIXL pull)
    batch_exists_v2                    local-DRAM residency + router-hint cover

Threading: KVCR owns a single daemon "progress thread" (``kvcr/progress.py``)
that solely holds the NIXL agent + ZMQ control socket and advances every
in-flight op. This backend never touches NIXL directly. Main-thread state
(residency, pins) is advanced by calling ``kvcr.poll_completed()``, which two
threads here do: the HiCache controller's prefetch thread (inside
``_drain_until``) and one daemon of our own, ``kvcr-source-pump``. The pump
exists because a *source*-side serve makes no progress unless somebody polls,
and an idle worker has no traffic of its own to poll for -- see
``_start_source_pump``. ``_poll_lock`` serializes the two.

Both directions of the KV path are now wired to real KVCR operations: set via
``deposit``, get via the unified ``deliver`` (which the core routes per key to
either its local DRAM tier or a source-peer NIXL pull, the latter gated on a
router hint registered for the request via ``submit_hint``). ``_drain_until``
blocks the calling thread until the op reports -- by design: that caller is the
controller's dedicated prefetch daemon, and ``_page_transfer`` reads
``completed_tokens`` as soon as it returns.

Both zero-copy call shapes are implemented: ``batch_*_v2`` (PoolTransfer, used
by HybridCacheController) and ``batch_*_v1`` (keys + host_indices, used by
HiRadixCache's ``_page_{get,set}_zero_copy``). v1 is a thin KV-pool wrapper
over v2, except that a bufferless hybrid anchor is a successful no-op. The
remaining DRAFT edges are the byte-copy legacy methods
(``get``/``set``/``batch_get``/``batch_set``), which no zero-copy backend uses.

Segment sub-blocking: a host KV page is not one contiguous run. MHA stores K
and V in separate halves of the pool tensor (and per-layer sub-runs in
``layer_first`` layout), so ``get_page_buffer_meta`` returns several
non-contiguous segments per page. KVCR's local tier copies exactly one
``MemDescriptor`` into a matching fixed-size arena slot, so each page deposits
as one block-key per physical component (page key + pool + component suffix).
The sizes and counts are discovered by probing each pool once at registration.
This ``page -> component-keys`` fan-out is a LOCAL-tier identity detail only;
the remote/source path (Workstream B) matches on router-hint page hashes through
``StrKeyHintAdapter``.

Hybrid pools are discovered and validated at registration finalization. Each
physical pool keeps its own buffer regions, component geometry, and key
namespace; the logical KV anchor is intentionally absent from that manifest.
Local deposit, existence, and delivery use KVCR's plural framework regions and
size-classed arenas. Hinted remote delivery uses the same physical descriptors;
the source validates every component's size against its destination before NIXL.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import socket
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

import msgspec
import torch
from kvcr import KVCR, KVCRBindings
from kvcr.config import (
    FrameworkDramInput,
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramInfo,
    RemoteFWDramOptions,
)
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.policy import (
    FIFOPolicy,
    G3FIFOPolicy,
    G3LRUPolicy,
    KVCachePolicy,
    LRUPolicy,
)
from kvcr.types import BlockKey, MemDescriptor, OpEntryStatus, QueryStatus

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.storage.kvcr.kvcr_config import (
    MAX_TCP_PORT,
    KVCRBackendConfig,
)
from sglang.srt.mem_cache.storage.kvcr.pin_adapter import NoFrameworkPinning
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    RouterHint,
    StrKeyHintAdapter,
)
from sglang.srt.mem_cache.storage.kvcr.router_hint import encode_key as _encode_key
from sglang.srt.utils import dynamic_import

logger = logging.getLogger(__name__)

# Backoff bounds for _drain_until's completion poll. Start tight so a local-tier
# hit (already resident, microseconds away) is not needlessly delayed, then back
# off so a remote NIXL fetch does not spin a core while the KVCR progress thread
# does the actual work.
_DRAIN_POLL_MIN_S = 50e-6
_DRAIN_POLL_MAX_S = 2e-3

# How often the source-side pump calls poll_completed() when this worker has no
# get/set traffic of its own. See _source_pump_func: a peer's fetch cannot make
# progress until we pump, so this bounds how long a P2P source keeps a peer
# waiting. Well under KVCR's operation_timeout_ms so a pin still has room to
# complete before the source-side deadline expires.
_SOURCE_PUMP_INTERVAL_S = 5e-3

# Consecutive pump faults tolerated before giving up. Arbitrary; chosen to ride
# out a blip without spinning on a core that is genuinely dead. At the interval
# above this is ~50ms of retrying.
_PUMP_MAX_CONSECUTIVE_FAULTS = 10

# How long close() waits for the pump to leave its last poll. One pump period is
# the wait plus one poll_completed(), so this is ample unless the poll itself has
# stalled -- which is precisely the case close() must not close underneath.
_PUMP_JOIN_TIMEOUT_S = 1.0

# How often the remote-path counters are summarized to the log. The remote path
# is the whole point of this backend and it fails *silently* -- a hint that
# never arrives and a fetch that returns nothing both look like an ordinary
# cache miss from outside the process. One line per interval is the cheapest
# way for an operator to tell "the router stopped hinting" from "the transfers
# are failing", which need fixes in different repositories.
_STATS_LOG_INTERVAL_S = 30.0

# The only transport KVCR's ZMQ control channel can dial a peer over, and the
# bind wildcards that are legal to bind but cannot be dialed. Loopback is *not*
# here: colocated workers are the normal single-host topology. A hint arrives
# from outside this process, so both are checked before the endpoint reaches
# ``submit_hint`` -- see _split_control_endpoint.
_CONTROL_SCHEME = "tcp://"
_UNDIALABLE_HINT_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*"})

# Hybrid physical keys changed once when pool qualification was introduced,
# again when cross-worker compatibility became part of their identity, and now
# when TP-replicated DeepSeek-V4 pools acquired one shared owner namespace.
_HYBRID_KEY_SCHEMA_VERSION = "v4"

_DEEPSEEK_V4_PHYSICAL_POOLS = frozenset(
    {
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
        PoolName.DEEPSEEK_V4_C128,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        PoolName.DEEPSEEK_V4_C128_STATE,
    }
)
_DEEPSEEK_V4_ALLOWED_POOLS = _DEEPSEEK_V4_PHYSICAL_POOLS | {PoolName.SWA}


@dataclass(frozen=True)
class _PoolContext:
    """Validated host-memory shape for one physical HiCache pool.

    ``component_sizes`` is the page-major order returned by that pool's own
    ``get_page_buffer_meta`` for one logical page. Equal-sized components share
    one KVCR local arena while retaining their own pool-qualified keys.
    """

    name: PoolName
    host_pool: HostKVCache
    buffers: Tuple[torch.Tensor, ...]
    regions: Tuple[FrameworkDramInput, ...]
    page_size: int
    component_sizes: Tuple[int, ...]


# KVCR core methods this backend calls. Checked once at startup because
# ``nvidia-kvcr`` is pre-1.0 and pinning its version would not help: the
# distribution has sat at 0.1.0 across renames that moved ``kvcc.kvcc`` to
# ``kvcc.api``, deleted ``nixl.py``, dropped ``has_pending_work``, changed
# ``query`` from returning statuses to ``(status, tier)`` pairs, and finally
# renamed the whole package from ``kvcc`` to ``kvcr`` -- still 0.1.0. A missing
# name surfaces here as one legible error naming what is absent, rather than as
# an AttributeError from inside a prefetch on the first cache miss.
_REQUIRED_KVCR_METHODS = (
    "deposit",
    "deliver",
    "discard_hint",
    "poll_completed",
    "query",
    "submit_hint",
)


def _require_kvcr_api() -> None:
    missing = [name for name in _REQUIRED_KVCR_METHODS if not hasattr(KVCR, name)]
    if missing:
        raise RuntimeError(
            f"KVCRStore: the installed nvidia-kvcr is missing {missing}. This "
            "backend tracks the kvcr core's current API; upgrade nvidia-kvcr "
            "or use a SGLang revision matching your kvcr."
        )


# Placement/eviction policies selectable by name from extra_config. Mirrors the
# same table on the vLLM side so a name means the same thing in both engines'
# configs and an A/B is comparable across them.
_BUILTIN_POLICIES: Dict[str, type] = {
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "g3_fifo": G3FIFOPolicy,
    "g3_lru": G3LRUPolicy,
}


def _resolve_policy(name: str) -> KVCachePolicy:
    """Build the policy named in extra_config.

    The core picks its own default when handed ``None``, and that default is not
    a stable interface -- it moved from FIFO to LRU in kvcc e3a816e. So this
    backend always names one, and the name is what gets logged and recorded
    alongside a benchmark number.
    """
    policy_type = _BUILTIN_POLICIES.get(name)
    if policy_type is None:
        if "." not in name:
            raise ValueError(
                f"KVCRStore: unknown policy {name!r}. Supported: "
                f"{sorted(_BUILTIN_POLICIES)}; an external policy must be given "
                "as a fully qualified module.Class path."
            )
        policy_type = dynamic_import(name)
        if not isinstance(policy_type, type) or not issubclass(
            policy_type, KVCachePolicy
        ):
            raise TypeError(f"KVCRStore: {name} is not a KVCachePolicy subclass")
    return policy_type()


# How often a fault escaping into the HiCacheStorage surface is logged with its
# traceback. The guard exists for repeatable faults (a peer that stays down, a
# core in a bad state), so one traceback per prefetch would be the loudest thing
# in the log while saying nothing new after the first.
_FAULT_LOG_INTERVAL_S = 30.0


def _fail_closed(on_error):
    """Never let an exception out of a ``HiCacheStorage`` entry point.

    HiCache's three storage threads (``prefetch_thread_func``,
    ``prefetch_io_aux_func``, ``backup_thread_func``) each catch only ``Empty``.
    Anything else ends the thread, and they are unsupervised daemons, so one
    exception disables L2 and L3 for the life of the process -- and does it
    silently, since every later request just reports a cache miss.

    It is not only a lost cache. Those loops are the only ones that give back
    what they reserved: ``prefetch_io_aux_func`` calls
    ``append_host_mem_release`` (without it ``prefetch_tokens_occupied`` climbs
    until the rate limiter blocks all prefetching, permanently), and
    ``backup_thread_func`` is the sole producer for ``ack_backup_queue`` (without
    it ``HiRadixCache`` never calls ``entry.release_host()`` and backed-up nodes
    pin host pages forever).

    So a fault degrades to "this batch missed": HiCache recomputes. That is safe
    before an asynchronous operation is accepted, and after it reports terminal.
    ``_submit_and_wait`` contains polling faults between those boundaries so this
    guard can never return a live operation's framework-owned pages as a miss.
    ``on_error`` builds the miss from the same arguments the method received,
    because a caller reads the shape of the result, not just its truthiness.

    Deliberately not applied to ``close()`` or ``register_mem_pool_host()``:
    those run on the scheduler thread during setup and teardown, where an
    exception is visible and worth surfacing rather than swallowing.
    """

    def decorate(method):
        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception:
                self._note_fault(method.__name__)
                return on_error(self, *args, **kwargs)

        return guarded

    return decorate


def _miss_per_transfer(self, transfers, *_args, **_kwargs) -> Dict[str, List[bool]]:
    """Every page of every transfer failed, keyed as the v2 callers expect."""
    return {str(t.name): [False] * len(t.keys or []) for t in transfers}


def _miss_per_key(self, keys, *_args, **_kwargs) -> List[bool]:
    return [False] * len(keys)


def _no_prefix(self, *_args, **_kwargs) -> PoolTransferResult:
    return PoolTransferResult.empty()


def _split_control_endpoint(endpoint: str) -> Optional[Tuple[str, str, int]]:
    """``(scheme_and_host, host, port)`` for a well-formed control endpoint.

    Splits on the *last* colon so a bracketed IPv6 literal
    (``tcp://[fd00::1]:25000``), whose address contains colons of its own,
    survives intact. Returns None for anything the control channel should not
    be handed: only ``tcp://`` is a ZMQ transport a peer can be dialed over,
    and only a real port number names a peer rather than a guess.
    """
    prefix, sep, port = endpoint.rpartition(":")
    if not sep or not port.isdigit():
        return None
    port_num = int(port)
    if not 1 <= port_num <= MAX_TCP_PORT:
        return None
    if not prefix.startswith(_CONTROL_SCHEME):
        return None
    host = prefix[len(_CONTROL_SCHEME) :]
    if not host or host in _UNDIALABLE_HINT_HOSTS:
        return None
    return prefix, host.strip("[]"), port_num


def _offset_endpoint_port(endpoint: str, offset: int) -> Optional[str]:
    """A validated ``tcp://host:port`` with ``offset`` added, or None."""
    split = _split_control_endpoint(endpoint)
    if split is None:
        return None
    prefix, _host, port = split
    if port + offset > MAX_TCP_PORT:
        return None
    return f"{prefix}:{port + offset}"


def _reject_unaddressable_parallelism(storage_config: HiCacheStorageConfig) -> None:
    """Refuse the parallel layouts whose pages this backend cannot tell apart.

    A KVCR block key is ``sha256(token ids)#<segment>``: it names the tokens and
    nothing about which slice of the model produced the bytes. So every rank
    coordinate that changes a page's *contents* has to be separated some other
    way, or two ranks holding different bytes agree on a key and a fetch returns
    the wrong KV with no error anywhere. ``_rank_port_offset`` separates
    ``(dp, attn_cp, attn_tp)`` by giving each rank its own control port and
    realigning every incoming hint onto it; the two below have no such
    separation. Mooncake, which keys into a shared store rather than a per-rank
    one, instead folds both into its key suffixes.

    - Pipeline parallelism: ``pp_rank`` is absent from the port offset, so
      ``pp0/tp0`` and ``pp1/tp0`` derive the same port. The loser of that bind
      is invisible (see :meth:`KVCRStore._control_port`), and a hint aimed at
      one rank reaches the other, filling the second half of the layers with
      the first half's KV.
    - Heterogeneous TP: ``should_split_heads`` says the deployment expects
      per-head-slice keys so a tp4 and a tp8 peer can share a prefix. This
      backend emits one key per page either way, so those peers agree on a key
      while holding different head slices.
    """
    if storage_config.pp_size > 1:
        raise RuntimeError(
            "KVCRStore does not support pipeline parallelism (pp_size="
            f"{storage_config.pp_size}). KVCR block keys carry no pp_rank, so "
            "two pipeline stages of one engine would share both a control port "
            "and a key namespace, and a peer fetch would return another stage's "
            "layers. Run with pp_size=1, or use a backend that namespaces keys "
            "per stage (e.g. mooncake)."
        )
    if storage_config.should_split_heads:
        raise RuntimeError(
            "KVCRStore does not support heterogeneous TP (tp_lcm_size="
            f"{storage_config.tp_lcm_size} > tp_size={storage_config.tp_size}). "
            "Head splitting exists so peers at different TP degrees can share a "
            "prefix, but KVCR block keys are page-level and carry no head slice, "
            "so those peers would agree on a key while holding different heads. "
            "Drop tp_lcm_size from the backend extra config, or use a backend "
            "with split-head support (e.g. mooncake)."
        )


def _dp_stride(storage_config: HiCacheStorageConfig) -> int:
    """How many schedulers one attention-DP rank of this engine owns.

    Also the port stride between two DP ranks, which is what the dynamo side has
    to multiply the DP rank by when it advertises one source endpoint per rank.
    """
    return storage_config.attn_cp_size * storage_config.tp_size


def _within_dp_offset(storage_config: HiCacheStorageConfig) -> int:
    """This rank's port offset *inside* its attention-DP group.

    Within one DP group, attention shards along ``(attn_cp, attn_tp)``, and a
    peer's shard is interchangeable with ours only when both coordinates match.
    So this is also the offset to apply to a source endpoint that the router has
    already resolved to the right DP rank.
    """
    return storage_config.attn_cp_rank * storage_config.tp_size + storage_config.tp_rank


def _source_rank_offset(storage_config: HiCacheStorageConfig) -> int:
    """Offset from the router's selected DP base to our matching source rank."""
    if not storage_config.tp_rank_is_attention_scoped:
        return storage_config.tp_rank
    return _within_dp_offset(storage_config)


def _rank_port_offset(storage_config: HiCacheStorageConfig) -> int:
    """This scheduler's port offset from the configured base port.

    Every ``(dp, attn_cp, attn_tp)`` rank of one engine runs its own KVCRStore in
    its own process on the same host, all reading the same ``extra_config``, so
    the offset has to be the full rank coordinate: ``tp_rank`` alone repeats once
    per DP group, and two ranks that pick the same port is invisible from the
    outside (see ``_control_port``).

    When ``tp_rank``/``tp_size`` are attention-scoped (``cache_controller``
    substitutes ``attn_tp_*``), SGLang lays ranks out as
    ``tp_rank = (dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size +
    attn_tp_rank`` (``compute_dp_attention_world_info``), which is exactly what
    is reassembled here. DSV4 round-robin CP has ``dp_size == 1`` and
    ``tp_size == 1``, so rank scope must be explicit rather than inferred from
    DP width. When ranks are not attention-scoped, ``tp_rank`` already is the
    engine-global coordinate and adding CP would double-count it.
    """
    if not storage_config.tp_rank_is_attention_scoped:
        return storage_config.tp_rank
    return storage_config.dp_rank * _dp_stride(storage_config) + _within_dp_offset(
        storage_config
    )


def _highest_rank_port_offset(storage_config: HiCacheStorageConfig) -> int:
    """The largest offset ``_rank_port_offset`` can return for this engine.

    Mirrors ``_rank_port_offset`` with every rank coordinate at its maximum, so
    the two must be edited together.
    """
    if not storage_config.tp_rank_is_attention_scoped:
        return storage_config.tp_size - 1
    return storage_config.dp_size * _dp_stride(storage_config) - 1


def _ephemeral_port() -> int:
    """Reserve an OS-assigned free TCP port and return it.

    There is an inherent bind-then-rebind race, but KVCR's control channel and
    NIXL listener are the only consumers and both bind immediately afterwards.
    """
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class KVCRStore(HiCacheStorage):
    """HiCacheStorage backend backed by the KVCR P2P coordinator (draft)."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool: Optional[HostKVCache] = None,
    ) -> None:
        _reject_unaddressable_parallelism(storage_config)
        self._storage_config = storage_config
        self._config = KVCRBackendConfig.from_extra_config(storage_config.extra_config)
        self.mem_pool_host = mem_pool

        # A per-worker unique NIXL agent name and control endpoint. Colocated
        # workers must not collide, so include the rank coordinate + a uuid. The
        # uuid also keeps a restarted rank from reusing a name its peers still
        # have in their remote-agent tables.
        self._agent_name = (
            f"kvcr-sgl-dp{storage_config.dp_rank}"
            f"-tp{storage_config.tp_rank}-{uuid.uuid4().hex[:8]}"
        )
        self._pinning = NoFrameworkPinning()
        self._key_hint_adapter = StrKeyHintAdapter()
        self.registered_pools: Dict[PoolName, HostKVCache] = {}
        self._pool_contexts: Dict[PoolName, _PoolContext] = {}
        self._logical_anchor = False
        self._compatibility_digest: Optional[str] = None
        self._local_dram_buffers: Dict[int, torch.Tensor] = {}

        # Single-pool layouts construct KVCR in register_mem_pool_host(). A
        # hybrid layout first presents a bufferless logical anchor, then its
        # physical pools, so construction is deferred to the registration
        # finalizer where the complete topology is available.
        self._kvcr: Optional[KVCR] = None
        self._control: Optional[ZmqPeerControlChannel] = None
        # Local DRAM slot geometry, learned by probing the host pool at
        # registration. slot_size == one page segment; segments_per_page ==
        # how many KVCR block-keys a single host page fans out into.
        self._slot_size: Optional[int] = None
        self._segments_per_page: Optional[int] = None
        # Completions drained from poll_completed() that belong to an op other
        # than the one currently being waited on. poll_completed() clears the
        # core's queue, so a result seen by the wrong waiter would be lost
        # without this stash. Guarded by _poll_lock -- the source pump drains
        # the same queue, so it can be the one to observe the completion of a get.
        self._completed_ops: Dict[int, Dict] = {}
        # Handles a _drain_until is currently blocked on. A completion for
        # anything else is dropped rather than stashed.
        #
        # A waiter remains live until the core reports terminal. KVCR cannot yet
        # abort/fence a remote write, so returning on a wall-clock deadline would
        # let HiCache reuse a destination the NIC may still mutate.
        self._waiting_ops: Set[int] = set()
        # Serializes poll_completed() between the prefetch thread (_drain_until)
        # and the source pump. poll_completed() both drains a queue and advances
        # core state machines, so two callers must not interleave.
        self._poll_lock = threading.Lock()
        self._pump_stop = threading.Event()
        self._pump_thread: Optional[threading.Thread] = None
        # Source of request ids for the core's hint table; see
        # _hint_request_id. Locked because HiCache runs one prefetch thread but
        # the v1 entry points are reachable from the scheduler thread too.
        self._hint_id_lock = threading.Lock()
        self._next_hint_id = 0
        # Remote-path counters, logged periodically by _note. Without them a
        # hinted get that returns nothing is indistinguishable from a request
        # the router never attached a hint to -- the two have opposite causes
        # (a broken fetch here vs. an index miss upstream) and the backend is
        # the only place that can tell them apart. Guarded by _stats_lock
        # because the v2 entry points run on the prefetch thread while the
        # source pump touches the same counters.
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = defaultdict(int)
        self._next_stats_log_at = 0.0
        # Separate clock from the stats line: a fault is rarer and carries a
        # traceback, so throttling the two together would let a chatty stats
        # interval swallow the first report of a fault.
        self._next_fault_log_at = 0.0

        if mem_pool is not None:
            self.register_mem_pool_host(mem_pool)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        if self._kvcr is not None:
            if mem_pool_host is self.mem_pool_host:
                return
            raise RuntimeError(
                "KVCRStore cannot replace the host-pool anchor after KVCR "
                "initialization."
            )
        super().register_mem_pool_host(mem_pool_host)
        if mem_pool_host.kv_buffer is None:
            self._logical_anchor = True
            return
        self._logical_anchor = False
        self._build_kvcr(mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name) -> None:
        """Record one physical pool before KVCR freezes its memory topology."""
        if self._kvcr is not None:
            existing = self.registered_pools.get(host_pool_name)
            if existing is host_pool or (
                host_pool_name == PoolName.KV and host_pool is self.mem_pool_host
            ):
                return
            raise RuntimeError(
                "KVCRStore cannot add or replace host pools after KVCR initialization."
            )
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

    def finalize_mem_pool_registration(self) -> None:
        """Construct KVCR only after the initial host-pool topology is known."""
        if self._kvcr is not None:
            return
        self._pool_contexts = self._collect_pool_contexts()
        if self.mem_pool_host is None:
            raise RuntimeError("KVCRStore cannot finalize without a host-pool anchor.")
        if self._logical_anchor:
            self._freeze_hybrid_key_namespace()
            self._build_hybrid_kvcr()
            return
        self._build_kvcr(self.mem_pool_host)

    @staticmethod
    def _iter_host_pool_buffers(host_pool: HostKVCache) -> Iterator[torch.Tensor]:
        """Yield every contiguous tensor a physical pool exposes for I/O."""
        get_buffers = getattr(host_pool, "get_hybrid_pool_buffer", None)
        raw_buffers = (
            get_buffers()
            if callable(get_buffers)
            else getattr(host_pool, "kv_buffer", None)
        )
        if raw_buffers is None:
            return
        if isinstance(raw_buffers, torch.Tensor):
            raw_buffers = (raw_buffers,)
        elif not isinstance(raw_buffers, (list, tuple)):
            raise TypeError(
                "KVCRStore host-pool buffers must be a tensor, list, or tuple; "
                f"got {type(raw_buffers).__name__}."
            )
        for buffer in raw_buffers:
            if buffer is None:
                continue
            if not isinstance(buffer, torch.Tensor):
                raise TypeError(
                    "KVCRStore host-pool buffer must be a torch.Tensor; "
                    f"got {type(buffer).__name__}."
                )
            if buffer.numel() <= 0:
                raise ValueError(
                    "KVCRStore physical host-pool buffers cannot be empty."
                )
            if not buffer.is_contiguous():
                raise ValueError(
                    "KVCRStore physical host-pool buffers must be contiguous."
                )
            yield buffer

    @staticmethod
    def _span_is_registered(
        address: int, length: int, regions: Tuple[FrameworkDramInput, ...]
    ) -> bool:
        if address < 0 or length <= 0:
            return False
        end = address + length
        return any(
            address >= region.address and end <= region.address + region.length
            for region in regions
        )

    def _inspect_physical_pool(
        self, name: PoolName, host_pool: HostKVCache
    ) -> Optional[_PoolContext]:
        """Describe one pool using its buffers and one-page metadata accessor."""
        buffers = tuple(self._iter_host_pool_buffers(host_pool))
        if not buffers:
            if (
                name == PoolName.KV
                and host_pool is self.mem_pool_host
                and self._logical_anchor
                and getattr(host_pool, "kv_buffer", None) is None
            ):
                return None
            raise RuntimeError(
                f"KVCRStore physical pool {name} exposes no host-memory buffers."
            )

        regions = tuple(
            FrameworkDramInput(
                address=int(buffer.data_ptr()),
                length=int(buffer.numel() * buffer.element_size()),
            )
            for buffer in buffers
        )
        page_size = int(getattr(host_pool, "page_size", 0) or 0)
        if page_size <= 0:
            raise RuntimeError(
                f"KVCRStore physical pool {name} has invalid page_size={page_size}."
            )
        probe_indices = torch.arange(page_size, dtype=torch.int64)
        try:
            meta = host_pool.get_page_buffer_meta(probe_indices)
        except Exception as exc:
            raise RuntimeError(
                f"KVCRStore could not inspect physical pool {name}."
            ) from exc
        if meta is None or len(meta) != 2:
            raise RuntimeError(
                f"KVCRStore physical pool {name} returned invalid page metadata."
            )
        ptr_list, size_list = meta
        if not ptr_list or len(ptr_list) != len(size_list):
            raise RuntimeError(
                f"KVCRStore physical pool {name} returned mismatched page metadata."
            )
        component_sizes = tuple(int(size) for size in size_list)
        for address, length in zip(ptr_list, component_sizes):
            if not self._span_is_registered(int(address), length, regions):
                raise RuntimeError(
                    f"KVCRStore physical pool {name} returned a page component "
                    "outside its exposed buffers."
                )
        return _PoolContext(
            name=name,
            host_pool=host_pool,
            buffers=buffers,
            regions=regions,
            page_size=page_size,
            component_sizes=component_sizes,
        )

    def _collect_pool_contexts(self) -> Dict[PoolName, _PoolContext]:
        """Build a deterministic manifest of all registered physical pools."""
        candidates = dict(self.registered_pools)
        if (
            self.mem_pool_host is not None
            and not self._logical_anchor
            and PoolName.KV not in candidates
        ):
            # The non-hybrid controller registers only its ordinary KV anchor.
            candidates[PoolName.KV] = self.mem_pool_host

        contexts: Dict[PoolName, _PoolContext] = {}
        for name in sorted(candidates, key=str):
            context = self._inspect_physical_pool(name, candidates[name])
            if context is not None:
                contexts[name] = context
        return contexts

    def _freeze_hybrid_key_namespace(self) -> None:
        """Bind physical component keys to this cache ABI and rank topology."""
        if not self._pool_contexts:
            raise RuntimeError(
                "KVCRStore cannot build a hybrid key namespace from an empty "
                "physical pool manifest."
            )
        cache_abi = self._config.cache_abi
        if self._config.enable_remote_hint and (not cache_abi or not cache_abi.strip()):
            raise RuntimeError(
                "KVCR hybrid remote hints require cache_abi in "
                "--hicache-storage-backend-extra-config. It must identify the "
                "model revision, cache dtype/quantization, and layer mapping."
            )
        logical_page_size = int(getattr(self.mem_pool_host, "page_size", 0) or 0)
        if self._config.enable_remote_hint and logical_page_size <= 0:
            raise RuntimeError(
                "KVCR hybrid remote hints require a positive logical anchor "
                "page_size for cache compatibility."
            )

        storage = self._storage_config
        if self._is_tp_replicated_dsv4():
            tp_identity = ("replicated", storage.tp_size)
        else:
            tp_identity = ("sharded", storage.tp_size, storage.tp_rank)
        identity = (
            _HYBRID_KEY_SCHEMA_VERSION,
            cache_abi,
            storage.is_page_first_layout,
            logical_page_size,
            tuple(
                (
                    str(name),
                    context.page_size,
                    context.component_sizes,
                )
                for name, context in sorted(
                    self._pool_contexts.items(), key=lambda item: str(item[0])
                )
            ),
            (storage.pp_size, storage.pp_rank),
            tp_identity,
            storage.tp_rank_is_attention_scoped,
            (storage.attn_cp_size, storage.attn_cp_rank),
            storage.is_mla_model,
            storage.tp_lcm_size,
            storage.should_split_heads,
        )
        digest = hashlib.sha256(msgspec.msgpack.encode(identity)).hexdigest()
        if self._compatibility_digest not in (None, digest):
            raise RuntimeError("KVCR hybrid key namespace changed after it was frozen.")
        self._compatibility_digest = digest

    def _is_tp_replicated_dsv4(self) -> bool:
        """Whether this manifest is the TP-replicated DeepSeek-V4 stack.

        DeepSeek-V4's compressed cache and SWA state are identical across TP
        ranks. HiCache therefore deposits them only from TP0, just as its
        Mooncake backend does. Kimi-style hybrid stacks are deliberately not
        inferred from ``is_mla_model`` alone because their Mamba state is
        TP-sharded and must keep rank-specific ownership.
        """
        pool_names = set(self._pool_contexts)
        return (
            self._storage_config.is_mla_model
            and self._logical_anchor
            and bool(pool_names & _DEEPSEEK_V4_PHYSICAL_POOLS)
            and pool_names <= _DEEPSEEK_V4_ALLOWED_POOLS
        )

    def _control_port(self) -> int:
        """Bind port for this rank's KVCR control channel.

        Every scheduler of one engine runs its own KVCRStore in its own process
        on the same host, all reading the same ``extra_config``. A configured
        port is therefore a *base* port that must be offset by rank -- without
        the offset, rank 1 binds the port rank 0 already holds. That failure is
        invisible from the outside: ``ZmqPeerControlChannel`` binds from the
        progress thread, so the engine still starts, still registers, and still
        advertises an endpoint; only peer fetches to the losing rank break.
        ``_rank_port_offset`` covers the whole rank coordinate, so DP ranks get
        disjoint port blocks rather than colliding on the same ``base +
        tp_rank``.

        Port 0 means "ask the OS", which already guarantees distinctness -- and
        offsetting an OS-assigned port would land on an arbitrary port belonging
        to someone else, so the offset applies only to configured ports. It is
        reachable only local-only: ``KVCRBackendConfig`` refuses port 0 together
        with ``enable_remote_hint``, because an OS-assigned port is known only
        inside this process and so cannot be registered for peers to dial.

        A base port near the top of the range would push high ranks past 65535,
        where ``bind`` fails on a rank the operator never named. The whole block
        is checked here rather than just this rank's port, so every rank of the
        engine fails at startup with the same message instead of the low ranks
        coming up and the high ones dying.
        """
        configured = int(self._config.control_port)
        if configured <= 0:
            return _ephemeral_port()
        offset = _rank_port_offset(self._storage_config)
        highest = configured + _highest_rank_port_offset(self._storage_config)
        if highest > MAX_TCP_PORT:
            raise ValueError(
                f"KVCR control_port {configured} leaves no room for this "
                f"engine's {highest - configured + 1} schedulers: the highest "
                f"rank would bind {highest}, above {MAX_TCP_PORT}. Lower "
                "control_port in --hicache-storage-backend-extra-config."
            )
        return configured + offset

    def _build_kvcr(self, mem_pool_host: HostKVCache) -> None:
        framework_dram = self._framework_dram_region(mem_pool_host)
        local_dram = self._local_dram_region(mem_pool_host)
        if framework_dram is None:
            # Every byte this backend moves has one end in the host KV pool:
            # deposit reads pages out of it, deliver writes pages into it. Both
            # ends go through NIXL -- the local tier's copy is a loopback
            # transfer to our own agent, not a memcpy -- and NIXL only accepts
            # descriptors inside a registered region. Starting without this
            # registration means every transfer names unregistered memory.
            raise RuntimeError(
                "KVCRStore: the host KV pool is not a single contiguous "
                "kv_buffer tensor, so it cannot be registered with NIXL as one "
                "region. Both directions of the KV path address that pool, so "
                "the backend cannot run against this pool layout (per-layer "
                "tensor lists need per-tensor registration -- not implemented)."
            )
        if local_dram is None:
            # The local DRAM tier is the backend's only storage: deposit writes
            # into it and every source-side serve reads out of it (this backend
            # offers no framework memory -- see pin_adapter). Without it KVCR
            # fails every deposit entry individually, which reads downstream as
            # "the cache never hits" rather than "the backend is unusable", so
            # refuse to start instead.
            raise RuntimeError(
                "KVCRStore: could not probe the host page layout, so KVCR's "
                "local DRAM tier cannot be sized. The backend has no storage "
                "without it."
            )

        self._start_kvcr(framework_dram=framework_dram, local_dram=local_dram)

    def _build_hybrid_kvcr(self) -> None:
        """Construct KVCR over the validated physical hybrid-pool manifest."""
        if not self._pool_contexts:
            raise RuntimeError(
                "KVCRStore: hybrid host-pool manifest has no data pools."
            )

        framework_regions: List[FrameworkDramInput] = []
        seen_regions: Set[Tuple[int, int]] = set()
        for context in self._pool_contexts.values():
            for region in context.regions:
                identity = (region.address, region.length)
                if identity not in seen_regions:
                    seen_regions.add(identity)
                    framework_regions.append(region)

        component_counts = Counter(
            size
            for context in self._pool_contexts.values()
            for size in context.component_sizes
        )
        component_count = sum(component_counts.values())
        composite_page_bytes = sum(
            size * count for size, count in component_counts.items()
        )
        if not framework_regions or not component_count or not composite_page_bytes:
            raise RuntimeError("KVCRStore: hybrid host-pool manifest is empty.")

        configured_slots = self._config.local_dram_slots
        if configured_slots > 0:
            page_capacity = configured_slots // component_count
            budget_name = "local_dram_slots"
            budget = configured_slots
        else:
            page_capacity = self._config.local_dram_bytes // composite_page_bytes
            budget_name = "local_dram_bytes"
            budget = self._config.local_dram_bytes
        if page_capacity <= 0:
            raise RuntimeError(
                f"KVCRStore: {budget_name}={budget} cannot hold one complete "
                f"hybrid page ({component_count} components, "
                f"{composite_page_bytes} bytes)."
            )

        local_arenas: List[LocalDramInfo] = []
        local_buffers: Dict[int, torch.Tensor] = {}
        for size, occurrences in sorted(component_counts.items()):
            slot_count = page_capacity * occurrences
            length = slot_count * size
            buffer = torch.empty(length, dtype=torch.uint8)
            local_buffers[size] = buffer
            local_arenas.append(
                LocalDramInfo(
                    address=buffer.data_ptr(),
                    length=length,
                    slot_count=slot_count,
                )
            )
        self._local_dram_buffers = local_buffers
        self._start_kvcr(
            framework_dram_regions=tuple(framework_regions),
            local_dram_arenas=tuple(local_arenas),
        )

    def _start_kvcr(
        self,
        *,
        framework_dram: Optional[FrameworkDramInput] = None,
        local_dram: Optional[LocalDramInfo] = None,
        framework_dram_regions: Tuple[FrameworkDramInput, ...] = (),
        local_dram_arenas: Tuple[LocalDramInfo, ...] = (),
    ) -> None:
        _require_kvcr_api()

        advertise = self._config.control_advertise_host or socket.gethostname()
        self._control = ZmqPeerControlChannel(
            self._config.control_host,
            self._control_port(),
            advertise,
        )

        # Give the NIXL listen socket a distinct ephemeral port per worker.
        nixl_listen_port = _ephemeral_port()

        config = KVCRConfig(
            nixl_agent_name=self._agent_name,
            enable_telemetry=self._config.enable_telemetry,
            operation_timeout_ms=self._config.operation_timeout_ms,
            nixl_listen_port=nixl_listen_port,
        )
        bindings = KVCRBindings(
            request_pin=self._pinning.request_pin,
            poll_pin_results=self._pinning.poll_pin_results,
            release_pin=self._pinning.release_pin,
            cancel_pin_request=self._pinning.cancel_pin_request,
            framework_control=self._control,
            key_hint_adapter=self._key_hint_adapter,
            policy=_resolve_policy(self._config.policy),
        )
        # eager_ctrl_connect / opportunistic_query / metadata_retry moved out of
        # KVCRConfig into the remote-forward-DRAM options in the wheel core.
        backend_configs = KVCRBackendConfigs(
            framework_dram=framework_dram,
            local_dram=local_dram,
            framework_dram_regions=framework_dram_regions,
            local_dram_arenas=local_dram_arenas,
            remote_fw_dram=RemoteFWDramOptions(
                eager_ctrl_connect=self._config.eager_ctrl_connect,
                opportunistic_query=self._config.opportunistic_query,
                metadata_retry_interval_ms=self._config.metadata_retry_interval_ms,
            ),
        )
        self._kvcr = KVCR(config, bindings, backend_configs)
        self._start_source_pump()
        logger.info(
            "KVCRStore initialized (agent=%s, slot_sizes=%s, remote_hint=%s, "
            "policy=%s)",
            self._agent_name,
            tuple(
                sorted(arena.length // arena.slot_count for arena in local_dram_arenas)
            )
            if local_dram_arenas
            else self._slot_size,
            self._config.enable_remote_hint,
            self._config.policy,
        )

    # ------------------------------------------------------------------
    # Source-side pump
    # ------------------------------------------------------------------

    def _start_source_pump(self) -> None:
        """Advance KVCR state even when this worker issues no traffic of its own.

        ``poll_completed()`` is what moves the core's state machines forward, and
        the only other caller is ``_drain_until`` -- which runs solely while
        *this* worker is doing a get or set. That is sufficient for the target
        side of a P2P fetch, but not for the source side: a peer's ``start_write``
        lands in the progress queue as a ``_SourcePinOp``, and until someone
        pumps, it is never pinned and never written. An otherwise idle worker
        would therefore serve nothing, and the requesting peer would sit until
        its deadline expired.

        So the pump is what makes a worker usable as a P2P *source*. It is
        deliberately a plain daemon thread rather than a scheduler-tick hook:
        the scheduler is free to be idle precisely when a peer needs us, and
        HiCacheStorage has no tick seam. Cost when nothing is in flight is one
        lock acquire plus an empty queue check per interval.
        """
        if not self._config.enable_remote_hint or self._pump_thread is not None:
            return
        self._pump_thread = threading.Thread(
            target=self._source_pump_func,
            name="kvcr-source-pump",
            daemon=True,
        )
        self._pump_thread.start()

    def _source_pump_func(self) -> None:
        """Poll until stopped, surviving transient faults.

        Exiting on the first exception would silently retire this worker as a
        P2P source for the rest of the process: nothing restarts the thread
        (``_start_source_pump`` returns early once ``_pump_thread`` is set), the
        engine keeps serving inference, and peers see only that we never have
        anything -- indistinguishable from a cold cache. A transient NIXL or ZMQ
        error is not worth that, so keep polling and count the faults.

        Consecutive failures are what distinguish a blip from a dead core. Give
        up only after ``_PUMP_MAX_CONSECUTIVE_FAULTS`` of them, and log that at
        error level, since past this point the worker is silently source-dead.
        """
        consecutive_faults = 0
        while not self._pump_stop.wait(_SOURCE_PUMP_INTERVAL_S):
            kvcr = self._kvcr
            if kvcr is None:
                return
            try:
                self._poll_once(kvcr)
            except Exception:
                consecutive_faults += 1
                self._note("source_pump_faults")
                if consecutive_faults >= _PUMP_MAX_CONSECUTIVE_FAULTS:
                    self._note("source_pump_dead")
                    logger.error(
                        "KVCRStore source pump failed %d times in a row; this "
                        "worker can no longer serve peers as a P2P source",
                        consecutive_faults,
                        exc_info=True,
                    )
                    return
                logger.warning(
                    "KVCRStore source pump failed (%d/%d consecutive)",
                    consecutive_faults,
                    _PUMP_MAX_CONSECUTIVE_FAULTS,
                    exc_info=True,
                )
            else:
                consecutive_faults = 0

    def _poll_once(self, kvcr: KVCR) -> None:
        """Drain one round of completions, stashing them for their waiters.

        Both the pump and ``_drain_until`` call this. Whoever gets there first
        drains the queue, so every result must be stashed rather than assumed to
        belong to the current caller -- except results for ops nobody is waiting
        on any more, which are dropped (see ``_waiting_ops``).
        """
        with self._poll_lock:
            for done_handle, entries in kvcr.poll_completed():
                if done_handle not in self._waiting_ops:
                    self._note("late_completions_dropped")
                    continue
                self._completed_ops[done_handle] = entries

    def _framework_dram_region(
        self, mem_pool_host: HostKVCache
    ) -> Optional[FrameworkDramInput]:
        """Contiguous engine host KV region, registered with NIXL, or None.

        Registering this region is what lets either direction of the KV path
        name a host page in a NIXL descriptor, and NIXL rejects descriptors
        outside a registered region. Both directions need it, including the
        ones that never leave this machine: ``deposit`` hands the core host
        pages as transfer *sources*, ``deliver`` hands it host pages as
        *destinations*, and KVCR's local tier moves both by submitting a
        transfer addressed to its own agent rather than by memcpy.

        None means "no single region covers the pool", which the caller turns
        into a startup failure -- there is no degraded mode to fall back to.

        For MHA and MLA the whole KV pool is a single contiguous ``kv_buffer``
        tensor (MHA's leading dim 2 packs the K and V halves; ``k_buffer`` /
        ``v_buffer`` are just views), so one (addr, length) covers it. Pools
        that keep ``kv_buffer`` as a per-layer *list* (e.g. DeepSeek V4
        paged/layer_first) are the genuine multi-tensor case: supporting them
        means registering each tensor separately, the way Mooncake's
        ``_iter_host_pool_buffers`` iterates them, which KVCR's single
        ``FrameworkDramInput`` has no shape for yet.
        """
        kv_buffer = mem_pool_host.kv_buffer
        if not isinstance(kv_buffer, torch.Tensor):
            return None
        address = kv_buffer.data_ptr()
        length = kv_buffer.numel() * kv_buffer.element_size()
        return FrameworkDramInput(address=address, length=length)

    def _local_dram_region(self, mem_pool_host: HostKVCache) -> Optional[LocalDramInfo]:
        """Allocate KVCR's own local DRAM tier (the buffer-only L3 pool).

        One slot holds one page *segment* (a K or V run of a page), so slot_size
        and the per-page segment count come from probing the pool's zero-copy
        meta -- see ``_probe_page_layout``. deposit() copies each segment into
        exactly one slot.
        """
        layout = self._probe_page_layout(mem_pool_host)
        if layout is None:
            logger.warning(
                "KVCRStore: could not probe host page layout; local DRAM tier "
                "disabled (DRAFT-STUB)."
            )
            return None
        segment_bytes, segments_per_page = layout
        self._slot_size = segment_bytes
        self._segments_per_page = segments_per_page

        slots = self._config.local_dram_slots
        if slots <= 0:
            slots = max(1, self._config.local_dram_bytes // segment_bytes)
        length = slots * segment_bytes

        # Anchor a contiguous host buffer for the slots and keep a reference so
        # it is not garbage-collected while NIXL has it registered.
        self._local_dram_buffer = torch.empty(length, dtype=torch.uint8)
        address = self._local_dram_buffer.data_ptr()
        return LocalDramInfo(address=address, length=length, slot_count=slots)

    def _probe_page_layout(
        self, mem_pool_host: HostKVCache
    ) -> Optional[tuple[int, int]]:
        """Learn ``(segment_bytes, segments_per_page)`` from the host pool.

        A host KV page is split into several non-contiguous segments (MHA: K/V
        halves, times per-layer runs in ``layer_first``). Rather than re-derive
        that layout arithmetic here, we ask the pool's own zero-copy accessor
        (``get_page_buffer_meta``) for a single page's segments and read the
        uniform segment byte size and the segment count straight off it.
        """
        page_size = mem_pool_host.page_size
        if not page_size:
            return None
        probe_indices = torch.arange(int(page_size), dtype=torch.int64)
        try:
            meta = mem_pool_host.get_page_buffer_meta(probe_indices)
        except Exception:
            logger.warning("KVCRStore: page-layout probe failed", exc_info=True)
            return None
        # Pools with no zero-copy support return None rather than a pair.
        if meta is None:
            return None
        ptr_list, size_list = meta
        if not ptr_list or not size_list:
            return None
        segment_bytes = int(size_list[0])
        if segment_bytes <= 0 or any(int(s) != segment_bytes for s in size_list):
            logger.warning(
                "KVCRStore: non-uniform host segment sizes %s; local tier disabled.",
                size_list,
            )
            return None
        return segment_bytes, len(ptr_list)

    def _locally_resident(self, segment_keys: List[BlockKey]) -> bool:
        """True iff KVCR's local DRAM tier holds every segment of a page.

        ``query`` is KVCR's own residency table, which is the only copy of that
        state: it moves keys to FILLING on deposit, to HIT on fill completion,
        and drops them on eviction, all inside the core. Mirroring it into a
        dict here would be a second copy that eviction can silently desync --
        and a stale "resident" answer is not a miss, it is a page we promise
        HiCache and then fail to deliver.

        Passing no request_id keeps this to residency only: a hint-covered key
        would otherwise report FETCHABLE, and the remote branch is the caller's
        to decide (see ``batch_exists_v2``).
        """
        if not segment_keys:
            return False
        statuses = list(self._kvcr.query(segment_keys))
        return len(statuses) == len(segment_keys) and all(
            status is QueryStatus.HIT for status, _tier in statuses
        )

    def close(self) -> None:
        """Stop the pump, then the core -- never the other way round.

        The pump calls ``poll_completed()``, which walks core state the core's
        own ``close()`` tears down (it closes the progress thread and the local
        tier). So a pump still inside a poll when the core goes away is a
        use-after-free on the KVCR side, not a benign late tick.

        The join therefore has to be honoured rather than merely attempted: if
        it times out, the pump is inside a poll that is taking longer than its
        entire interval, and closing under it is exactly the race. Leaving the
        core open leaks it for the remaining life of a process that is shutting
        down anyway, which is the strictly safer of the two outcomes.

        The core's own ``close()`` makes the same trade one level down: when its
        progress loop does not go quiescent it keeps the backend resources and
        raises, precisely so nothing unmaps memory a native transfer still
        references. We retain the reference and propagate that refusal. The
        owner must not clear the backend or destroy its registered host pools
        unless this method returns successfully.
        """
        self._pump_stop.set()
        pump = self._pump_thread
        if pump is not None:
            # Generous relative to the pump's own period: one poll plus slack.
            pump.join(timeout=_PUMP_JOIN_TIMEOUT_S)
            if pump.is_alive():
                logger.error(
                    "KVCRStore: source pump still running after %.1fs; leaving "
                    "the KVCR core open rather than closing it underneath a "
                    "live poll.",
                    _PUMP_JOIN_TIMEOUT_S,
                )
                raise RuntimeError(
                    "KVCR source pump did not stop; KVCR core remains open"
                )
            self._pump_thread = None
        if self._kvcr is not None:
            try:
                self._kvcr.close()
            except BaseException as error:
                # Keep the core reachable so process reclaim, rather than an
                # unsafe framework unmap, owns the remaining native resources.
                logger.exception(
                    "KVCRStore: KVCR core did not close cleanly; keeping the "
                    "core so its still-registered memory is not unmapped."
                )
                raise RuntimeError(
                    "KVCR core did not close; registered memory remains owned"
                ) from error
            self._kvcr = None

    # ------------------------------------------------------------------
    # v2 interface (the real HiCache path)
    # ------------------------------------------------------------------

    def _is_supported_transfer(self, transfer: PoolTransfer) -> bool:
        """Whether this core registered the memory named by ``transfer``.

        Ordinary models use the legacy KV pool. A logical-anchor model instead
        transfers only the physical pools captured in ``_pool_contexts``; the
        anchor itself is handled as a successful no-op by the v1 wrappers.
        """
        if not self._logical_anchor and transfer.name == PoolName.KV:
            return True
        if self._logical_anchor and transfer.name in self._pool_contexts:
            return True
        self._note(f"rejected_pool_{transfer.name}")
        return False

    @_fail_closed(_miss_per_transfer)
    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        """Offload host KV into KVCR's local DRAM tier via deposit()."""
        results: Dict[str, List[bool]] = {}
        if self._kvcr is None:
            return {str(t.name): [False] * len(t.keys or []) for t in transfers}
        for transfer in transfers:
            if not self._is_supported_transfer(transfer):
                results[str(transfer.name)] = [False] * len(transfer.keys or [])
                continue
            results[str(transfer.name)] = self._deposit_transfer(transfer)
        return results

    def _segment_key(
        self,
        page_key: str,
        seg: int,
        pool_name: PoolName = PoolName.KV,
    ) -> BlockKey:
        """KVCR block identity for one segment of a host page.

        A page fans out into ``segments_per_page`` KVCR blocks; the ``#<seg>``
        suffix keeps them distinct in the local tier. This identity is
        understood by both the local and remote paths. Router hints still name
        whole pages; their adapter strips everything after the first ``#``.

        The existing KV spelling is retained for single-pool and mixed-version
        compatibility. Physical hybrid pools add a versioned compatibility and
        pool namespace so equal page hashes cannot alias another model, rank,
        layout, pool, or component.
        """
        if pool_name == PoolName.KV:
            return _encode_key(f"{page_key}#{seg}")
        if self._compatibility_digest is None:
            raise RuntimeError(
                "KVCR hybrid component key requested before the physical pool "
                "manifest was finalized."
            )
        return _encode_key(
            f"{page_key}#{_HYBRID_KEY_SCHEMA_VERSION}/"
            f"{self._compatibility_digest}/{pool_name}/{seg}"
        )

    def _page_segment_keys(
        self,
        page_key: str,
        pool_name: PoolName = PoolName.KV,
        segments_per_page: Optional[int] = None,
    ) -> List[BlockKey]:
        if segments_per_page is None:
            context = self._pool_contexts.get(pool_name)
            segments_per_page = (
                len(context.component_sizes)
                if context is not None
                else (self._segments_per_page or 0)
            )
        return [
            self._segment_key(page_key, seg, pool_name)
            for seg in range(segments_per_page)
        ]

    def _deposit_transfer(self, transfer: PoolTransfer) -> List[bool]:
        keys = transfer.keys or []
        if not keys:
            return [False] * len(keys)
        # Build one source descriptor per (page, segment).
        built = self._host_descriptors(transfer)
        if built is None:
            logger.warning(
                "KVCRStore deposit skipped: no host descriptors for %d pages",
                len(keys),
            )
            return [False] * len(keys)
        descriptors, per_page_keys = built

        op_handle, result_map = self._submit_and_wait(
            lambda: self._kvcr.deposit(descriptors)
        )
        missing = len(descriptors) - len(result_map)
        failed = sum(1 for ok in result_map.values() if not ok)
        if failed or missing:
            # HiCache only reports "N pages failed", which cannot distinguish a
            # rejected deposit from a segment KVCR never reported on at all.
            logger.warning(
                "KVCRStore deposit op=%s: %d/%d segments failed, %d unreported "
                "(pages=%d)",
                op_handle,
                failed,
                len(descriptors),
                missing,
                len(keys),
            )

        # A page is stored iff every one of its segments landed. Nothing is
        # recorded on our side: the copy is now in KVCR's own slots, and its
        # residency table is what ``_locally_resident`` and the source path both
        # read. ``descriptors`` names the *host* pages we copied out of, which
        # HiCache is free to reuse the moment this call returns.
        results = [
            all(result_map.get(seg_key, False) for seg_key in page_keys)
            for page_keys in per_page_keys
        ]
        # Counted because the first question about any missed P2P fetch is
        # whether the source ever held the blocks, and until now every counter
        # here was on the get side -- so a source that quietly stored nothing
        # looked exactly like a target that quietly fetched nothing.
        self._note("deposit_pages_offered", len(keys))
        self._note("deposit_pages_stored", sum(results))
        return results

    def _host_descriptors(
        self, transfer: PoolTransfer
    ) -> Optional[Tuple[Dict[BlockKey, MemDescriptor], List[List[BlockKey]]]]:
        """Map each page key's segments to per-segment source MemDescriptors.

        Returns ``(descriptors, per_page_keys)``, or None if the pool meta can't
        be lined up with the requested keys. ``descriptors`` is the flat
        ``{component_key: MemDescriptor}`` mapping KVCR takes. For today's KV
        path every component is exactly ``slot_size`` bytes. Hybrid contexts
        instead preserve the component sizes their own pool reported so KVCR
        can select the matching local arena. ``per_page_keys`` groups the same
        keys by page so result folding cannot accidentally use a different key
        spelling.
        """
        host_indices = transfer.host_indices
        keys = transfer.keys or []
        if host_indices is None or not keys:
            return None

        context = self._pool_contexts.get(transfer.name)
        if context is not None:
            host_pool = context.host_pool
            component_sizes = context.component_sizes
            regions = context.regions
        elif transfer.name == PoolName.KV and self.mem_pool_host is not None:
            # Compatibility for direct single-pool users that have not called
            # the controller's registration finalizer yet.
            if self._segments_per_page is None or self._slot_size is None:
                return None
            host_pool = self.mem_pool_host
            component_sizes = (self._slot_size,) * self._segments_per_page
            regions = ()
        else:
            logger.warning(
                "KVCRStore: no physical pool context for transfer %s",
                transfer.name,
            )
            return None
        try:
            ptr_list, size_list = host_pool.get_page_buffer_meta(host_indices)
        except Exception:
            logger.warning(
                "KVCRStore: get_page_buffer_meta failed for pool %s",
                transfer.name,
                exc_info=True,
            )
            return None
        segments = len(component_sizes)
        if len(ptr_list) != len(size_list) or len(ptr_list) != len(keys) * segments:
            logger.warning(
                "KVCRStore: pool %s page meta count ptrs=%d sizes=%d != "
                "keys %d * components %d; layout changed since registration?",
                transfer.name,
                len(ptr_list),
                len(size_list),
                len(keys),
                segments,
            )
            return None
        descriptors: Dict[BlockKey, MemDescriptor] = {}
        per_page_keys: List[List[BlockKey]] = []
        for page_idx, key in enumerate(keys):
            base = page_idx * segments
            page_keys: List[BlockKey] = []
            for seg in range(segments):
                ptr = int(ptr_list[base + seg])
                size = int(size_list[base + seg])
                expected_size = component_sizes[seg]
                if size != expected_size:
                    logger.warning(
                        "KVCRStore: pool %s component %d size %d != registered size %d",
                        transfer.name,
                        seg,
                        size,
                        expected_size,
                    )
                    return None
                if context is not None and not self._span_is_registered(
                    ptr, size, regions
                ):
                    logger.warning(
                        "KVCRStore: pool %s component %d lies outside its "
                        "registered framework regions",
                        transfer.name,
                        seg,
                    )
                    return None
                segment_key = self._segment_key(key, seg, transfer.name)
                page_keys.append(segment_key)
                descriptors[segment_key] = MemDescriptor(
                    end_point_name=self._agent_name,
                    mem_type="DRAM",
                    addr=ptr,
                    size=size,
                    device_Id=0,
                    info="",
                )
            per_page_keys.append(page_keys)
        return descriptors, per_page_keys

    @_fail_closed(_miss_per_transfer)
    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        """Load KV into host memory via KVCR ``deliver()``.

        ``deliver`` is a single unified entry: the core auto-routes each block
        key by residency (see ``KVCR.deliver``). A key that ``deposit`` made
        locally resident is served from KVCR's own DRAM tier; a key that is only
        covered by this request's router hint is pulled from the source peer
        over NIXL. We hand ``deliver`` the *host page* segment descriptors as
        write destinations, so both paths land straight in the engine KV pool.

        The remote branch is gated on a well-formed hint having been registered
        with the core for this request_id (via ``submit_hint``); without one the
        core reports MISS for non-resident keys and we return them as failures,
        letting HiCache fall back to recompute. Hybrid pools use one
        request-scoped hint across every physical transfer in this call. A
        missing or malformed hint still permits local hits, while uncovered
        nonlocal components fail closed in the KVCR core.
        """
        results: Dict[str, List[bool]] = {}
        if self._kvcr is None:
            return {str(t.name): [False] * len(t.keys or []) for t in transfers}

        request_id = self._register_hint(extra_info)
        try:
            for transfer in transfers:
                if not self._is_supported_transfer(transfer):
                    results[str(transfer.name)] = [False] * len(transfer.keys or [])
                    continue
                results[str(transfer.name)] = self._deliver_transfer(
                    transfer, request_id
                )
        finally:
            self._discard_hint(request_id)
        return results

    def _discard_hint(self, request_id: Optional[str]) -> None:
        """Unregister the request's hint, without letting that fail the batch.

        Two reasons this is not a bare call in the ``finally``. It runs on the
        exception path too, where raising would replace the original fault with
        a less informative one; and on the success path a raise would discard a
        result set whose pages are already in host memory, turning a completed
        fetch into a recompute.

        The leak it cannot prevent is the core's: a hint we failed to discard
        stays in KVCR's request-scoped table. That is bounded per request and
        harmless to correctness (a stale entry only ever names a source we would
        have consulted anyway), so it is counted rather than retried.
        """
        if request_id is None:
            return
        try:
            self._kvcr.discard_hint(request_id)
        except Exception:
            self._note_fault("discard_hint")

    def _register_hint(
        self, extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> Optional[str]:
        """Parse a router hint and register it with the core for this request.

        Returns the request_id the core keys the hint on, or None when no
        well-formed hint is present / remote hints are disabled. The hint is
        submitted with an empty key set so the facade only records the
        advisory routing entry (and, if eager, warms the control connection) --
        the actual pull is driven by ``deliver`` in ``_deliver_transfer``, which
        lets us target engine host pages rather than KVCR-owned slots.
        """
        hint = self._parse_hint(extra_info)
        if hint is None:
            self._note("get_without_hint")
            return None
        self._note("get_with_hint")
        self._note("hinted_blocks", len(hint.block_hashes))
        # KVCR keys the request-scoped hint table on request_id; the controller
        # does not thread one through extra_info, so we mint our own.
        request_id = self._hint_request_id()
        self._kvcr.submit_hint(
            (),
            src=hint.source_control_endpoint,
            hints=hint,
            request_id=request_id,
        )
        logger.debug(
            "KVCRStore: registered router hint (source=%s, %d blocks, req=%s)",
            hint.source_control_endpoint,
            len(hint.block_hashes),
            request_id,
        )
        return request_id

    def _parse_hint(
        self, extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> Optional[RouterHint]:
        """The request's router hint, with its source endpoint aligned to us.

        The endpoint on the hint already names the right *DP rank* of the source:
        dynamo's router indexes workers as ``(worker_id, dp_rank)`` and resolves
        the source's advertised per-DP-rank map down to one endpoint before the
        hint ships. What it cannot resolve is the rank *within* that DP group --
        it has no TP concept, so the port it names is that DP rank's first
        scheduler. For a sharded cache, rank ``i`` of our DP group must pull from
        rank ``i`` of the source's. DeepSeek-V4 is the exception: its physical
        cache pools are TP-replicated and HiCache deposits them only from TP0,
        so every target TP rank pulls from TP0 of its matching CP slice.
        Realigning here mirrors what :meth:`_control_port` does on the bind side,
        with only the within-DP part of the offset since the DP part is the
        router's to apply.

        Getting this wrong does not fail -- KVCR block keys are token hashes and
        carry no rank identity, so a rank that dials the wrong peer receives a
        shard it will happily accept and decode from. Correctness therefore rests
        entirely on this offset, which is why an endpoint we cannot realign drops
        the hint (costing a recompute) rather than passing it through.

        The endpoint is an address this process will connect out to, taken from
        request-scoped data, so ``_offset_endpoint_port`` also decides whether it
        is dialable at all: transport, port range, and bind wildcards. What that
        cannot decide is whether the *named peer* is one we should trust, since
        nothing in the hint is authenticated. ``kv_hints`` is documented as
        router-set and never client-set; enforcing that is the ingress's job
        (SGLang RFC #36224), not reconstructible here.
        """
        if not self._config.enable_remote_hint:
            return None
        hint = RouterHint.maybe_from_extra_info(extra_info)
        if hint is None:
            return None
        if self._is_tp_replicated_dsv4():
            # Dynamo's endpoint names TP0/CP0 of the selected source DP rank.
            # Keep the CP slice, but route every TP replica to that slice's TP0
            # owner instead of to an empty same-rank KVCR instance.
            offset = 0
            if self._storage_config.tp_rank_is_attention_scoped:
                offset = (
                    self._storage_config.attn_cp_rank * self._storage_config.tp_size
                )
        else:
            offset = _source_rank_offset(self._storage_config)
        endpoint = _offset_endpoint_port(hint.source_control_endpoint, offset)
        if endpoint is None:
            logger.warning(
                "KVCRStore: dropping router hint, cannot align source endpoint "
                "%s to within-DP rank offset %d",
                hint.source_control_endpoint,
                offset,
            )
            return None
        return msgspec.structs.replace(hint, source_control_endpoint=endpoint)

    def _note_entry_statuses(self, entries: Dict) -> None:
        """Count how KVCR classified each block key, not just pass/fail.

        ``OpEntryResult.success`` is ``status is SUCCESS``, so a block the
        *policy* declined (``DROPPED``, returned when ``decide_ingest`` answers
        DROP) and a block that genuinely broke (``FAILED``) collapse into the
        same falsy value everywhere downstream. Both are correct to treat as
        "not stored" -- but they are not the same event to a reader: a rising
        DROPPED count means the local tier is under capacity pressure and the
        policy is doing its job, while a rising FAILED count means something is
        wrong. Keeping them apart is what makes the counters usable as evidence
        when a policy is being tuned, which the fault-injection run relies on.
        """
        dropped = failed = 0
        for entry in entries.values():
            if entry.status is OpEntryStatus.DROPPED:
                dropped += 1
            elif entry.status is not OpEntryStatus.SUCCESS:
                failed += 1
        if dropped:
            self._note("entries_dropped_by_policy", dropped)
        if failed:
            self._note("entries_failed", failed)

    def _note(self, event: str, count: int = 1) -> None:
        """Count a remote-path event, and summarize periodically at INFO.

        Per-event logging is not an option on this path -- it runs per prefetch,
        which is per request -- but silence is worse: when the remote path stops
        working there is nothing in any log to say so, and the first symptom is
        a throughput number nobody can attribute. So counters accumulate and one
        line goes out every ``_STATS_LOG_INTERVAL_S``, only while something is
        happening (an idle worker stays quiet because nothing increments).
        """
        with self._stats_lock:
            self._stats[event] += count
            now = time.monotonic()
            if now < self._next_stats_log_at:
                return
            self._next_stats_log_at = now + _STATS_LOG_INTERVAL_S
            snapshot = dict(self._stats)
        logger.info(
            "KVCRStore remote path (cumulative): %s",
            " ".join(f"{name}={value}" for name, value in sorted(snapshot.items())),
        )

    def _note_fault(self, method_name: str) -> None:
        """Record a fault the ``_fail_closed`` guard caught, and log it sparsely.

        Counted per entry point, not in aggregate: a fault in ``batch_set_v2``
        means offload is failing while reads may be fine, and the two are
        repaired in different places. The traceback goes out at most once per
        ``_FAULT_LOG_INTERVAL_S`` because the faults this guard is for repeat
        every prefetch -- but the *first* one is logged immediately, since a
        counter alone would not say what broke.
        """
        self._note(f"faults_{method_name}")
        with self._stats_lock:
            now = time.monotonic()
            if now < self._next_fault_log_at:
                return
            self._next_fault_log_at = now + _FAULT_LOG_INTERVAL_S
        logger.warning(
            "KVCRStore: %s failed; reporting a miss so HiCache recomputes. "
            "Repeated faults are counted in stats() as faults_%s.",
            method_name,
            method_name,
            exc_info=True,
        )

    def stats(self) -> Dict[str, int]:
        """Snapshot of the remote-path counters, for tests and for operators."""
        with self._stats_lock:
            return dict(self._stats)

    def _hint_request_id(self) -> str:
        """Fresh id scoping one ``batch_get_v2`` call's hint in the core.

        Must be unique per *call*, not per prefix. The id keys the core's
        request-scoped hint table, and this call ends by unregistering it
        (``discard_hint`` in the ``finally``) -- so two concurrent calls sharing
        an id means the first one to finish revokes the hint the second is
        still fetching against, which downgrades it to a silent local-only miss.

        Deriving the id from the hint content instead is what made that
        collision reachable: two requests sharing a prefix is the normal case
        here, not a rare one, and it is exactly when both want the same source.
        A counter has no such structure. It is process-local, which is all the
        core requires -- the hint table lives in this worker.
        """
        with self._hint_id_lock:
            self._next_hint_id += 1
            return f"kvcr-get-{self._next_hint_id}"

    def _deliver_transfer(
        self,
        transfer: PoolTransfer,
        request_id: Optional[str],
        *,
        local_only: bool = False,
    ) -> List[bool]:
        """Pull one transfer's pages into host memory via ``deliver``.

        Builds a ``{segment_key: host destination descriptor}`` map (the same
        page->segment fan-out as deposit) and issues a single ``deliver``. A
        page counts as loaded only when every one of its segments succeeded.
        """
        keys = transfer.keys or []
        if not keys:
            return [False] * len(keys)
        built = self._host_descriptors(transfer)
        if built is None:
            return [False] * len(keys)
        destinations, per_page_keys = built

        eligible_pages = [True] * len(per_page_keys)
        if local_only:
            request_id = None
            eligible_pages = [
                self._locally_resident(page_keys) for page_keys in per_page_keys
            ]
            destinations = {
                segment_key: destinations[segment_key]
                for eligible, page_keys in zip(eligible_pages, per_page_keys)
                if eligible
                for segment_key in page_keys
            }
            if not destinations:
                return [False] * len(keys)

        op_handle, result_map = self._submit_and_wait(
            lambda: self._kvcr.deliver(destinations, request_id=request_id)
        )

        results = [
            eligible and all(result_map.get(seg_key, False) for seg_key in page_keys)
            for eligible, page_keys in zip(eligible_pages, per_page_keys)
        ]
        loaded = sum(results)
        self._note("pages_requested", len(results))
        self._note("pages_loaded", loaded)
        if request_id is not None:
            # Separate from pages_loaded: a hinted deliver that lands nothing is
            # the failure this backend exists to make visible, and it is not the
            # same event as a local-tier miss on an unhinted request.
            self._note("hinted_pages_requested", len(results))
            self._note("hinted_pages_loaded", loaded)
            if getattr(self._config, "enable_telemetry", False):
                completed_components = [
                    key for key in destinations if result_map.get(key, False)
                ]
                requested_bytes = sum(
                    int(descriptor.size) for descriptor in destinations.values()
                )
                completed_bytes = sum(
                    int(destinations[key].size) for key in completed_components
                )
                result = (
                    "success"
                    if loaded == len(results)
                    else "partial"
                    if completed_components
                    else "failed"
                )
                logger.info(
                    "KVCRStore hinted transfer op=%s request=%s pool=%s "
                    "result=%s pages=%d/%d components=%d/%d bytes=%d/%d",
                    op_handle,
                    request_id,
                    transfer.name,
                    result,
                    loaded,
                    len(results),
                    len(completed_components),
                    len(destinations),
                    completed_bytes,
                    requested_bytes,
                )
        return results

    @_fail_closed(_no_prefix)
    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Longest available prefix: locally resident, else remote via hint.

        A page is available when either (a) all its segments are resident in
        KVCR's local DRAM tier, or (b) it is covered by this request's router
        hint (a peer holds it and ``batch_get_v2`` can pull it). The prefix is
        root-aligned and contiguous, so it stops at the first page that is
        neither. This gate is what makes the remote branch of ``batch_get_v2``
        reachable -- the controller only issues gets for the prefix reported
        here.
        """
        if self._logical_anchor:
            return self._batch_exists_hybrid(keys, pool_transfers or [], extra_info)
        if self._kvcr is None or self._segments_per_page is None:
            return PoolTransferResult.empty()
        # A sidecar pool this backend cannot serve makes the whole prefix
        # unusable, so report none rather than a KV-only prefix the caller would
        # read as covering every pool.
        if any(not self._is_supported_transfer(t) for t in pool_transfers or []):
            return PoolTransferResult.empty()
        # Same parse as batch_get_v2, so a hint this rank cannot align is
        # reported unavailable here rather than promised and then missed.
        hint = self._parse_hint(extra_info)
        prefix = 0
        remote_prefix = 0
        for key in keys:
            local = self._locally_resident(self._page_segment_keys(key))
            if not local:
                if not (hint is not None and hint.covers(key)):
                    break
                remote_prefix += 1
            prefix += 1
        # This gate is where the remote path is won or lost: the controller only
        # issues gets for the prefix reported here, so a hint that covers
        # nothing produces no get at all and leaves no other trace.
        self._note("exists_calls")
        if hint is not None:
            self._note("exists_with_hint")
            if remote_prefix:
                self._note("exists_hint_covered_pages", remote_prefix)
            else:
                self._note("exists_hint_covered_nothing")
        return PoolTransferResult(prefix, {})

    def _batch_exists_hybrid(
        self,
        keys: List[str],
        pool_transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo],
    ) -> PoolTransferResult:
        """Fold local or hinted components into valid logical-prefix endpoints.

        The KV anchor is virtual, so every input page starts as a candidate.
        Each physical pool then removes endpoints it cannot restore. A router
        hint is advisory at this stage; delivery still verifies every component
        and folds a missing or incompatible source entry into a page miss.
        """
        if self._kvcr is None:
            return PoolTransferResult.empty()

        hint = self._parse_hint(extra_info)
        kv_pages = len(keys)
        hit_count = {PoolName.KV: kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))
        for transfer in pool_transfers:
            if not restorable:
                break
            if not self._is_supported_transfer(transfer):
                return PoolTransferResult.empty()

            page_exists = [
                self._locally_resident(self._page_segment_keys(page_key, transfer.name))
                or (hint is not None and hint.covers(page_key))
                for page_key in keys
            ]
            boundary = 0
            pool_restorable: List[int] = []
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                try:
                    boundary = page_exists.index(False)
                except ValueError:
                    boundary = kv_pages
                pool_restorable = list(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    start = max(0, prefix_len - trailing)
                    if all(page_exists[start:prefix_len]):
                        pool_restorable.append(prefix_len)
                        if boundary == 0:
                            boundary = prefix_len
            else:
                raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")
            if boundary:
                hit_count[transfer.name] = boundary
            pool_restorable_set = set(pool_restorable)
            restorable = [
                prefix for prefix in restorable if prefix in pool_restorable_set
            ]

        final_pages = restorable[-1] if restorable else 0
        return PoolTransferResult(final_pages, hit_count, restorable)

    # ------------------------------------------------------------------
    # Progress pump
    # ------------------------------------------------------------------

    def _submit_and_wait(self, submit: Callable[[], int]) -> Tuple[int, Dict]:
        """Issue one KVCR op and block for its result, as ``(handle, results)``.

        ``submit`` runs under ``_poll_lock`` so the op is registered as awaited
        before anyone can drain its completion. Registering afterwards would
        race: a local-tier deposit can finish in microseconds while the source
        pump polls every 5 ms, so the pump would see a completion with no waiter,
        drop it as late, and the caller would wait forever for an op that
        actually succeeded.

        The handle comes back because it is the only join between our logs and
        KVCR's -- a failure here is usually diagnosed from the core's side.
        """
        with self._poll_lock:
            op_handle = submit()
            self._waiting_ops.add(op_handle)
        return op_handle, self._drain_until(op_handle)

    def _register_waiter(self, op_handle: int) -> None:
        """Claim ``op_handle``'s completion, so ``_poll_once`` stashes it.

        Idempotent: ``_submit_and_wait`` already did this under the lock it held
        across the submit, and ``_drain_until`` repeats it to cover a direct
        call.
        """
        with self._poll_lock:
            self._waiting_ops.add(op_handle)

    def _drain_until(self, op_handle: int, timeout_s: Optional[float] = None) -> Dict:
        """Pump kvcr.poll_completed() until ``op_handle`` reports terminal.

        Blocking here is the contract, not a compromise: this runs on the HiCache
        controller's dedicated ``prefetch_io_aux_func`` daemon thread, and
        ``_page_transfer`` inspects ``operation.completed_tokens`` immediately
        after ``page_get_func`` returns -- so results must be in hand by then.
        The scheduler thread is never involved; it only observes the resulting
        ``completed_tokens`` via the existing ``check_prefetch_progress`` tick.

        KVCR's transfer progress is owned by its own "kvcr-progress" daemon
        thread, which appends finished ops to a completion queue;
        ``poll_completed`` is a non-blocking drain of that queue with no
        condition variable to wait on. So we poll -- but yield between attempts
        (backing off to ``_DRAIN_POLL_MAX_S``) instead of spinning a bare loop,
        which otherwise burns a core and starves the progress thread that we are
        waiting on. Completions for other in-flight ops are stashed, never
        dropped.

        ``timeout_s`` is a soft observability threshold, not an ownership
        deadline. The adapter has no successful abort/quiescence result it can
        use today, so returning before KVCR reports terminal would let HiCache
        recycle a host page that an old transfer can still overwrite. An overdue
        peer therefore stalls this one prefetch thread until the core reports. A
        bounded fallback requires a KVCR terminal/quiescence contract or a
        KVCR-owned staging destination; it cannot be implemented safely in this
        adapter alone.
        """
        timeout = self._config.get_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        sleep_s = _DRAIN_POLL_MIN_S
        overdue = False
        poll_faulted = False
        try:
            self._register_waiter(op_handle)
            while True:
                # Always go through the stash: the source pump drains the same
                # queue, so our own completion may well be observed by it rather
                # than by the poll below.
                try:
                    self._poll_once(self._kvcr)
                except Exception:
                    # Submission already handed the core ownership of the
                    # transfer descriptors. A poll fault cannot prove that the
                    # transfer is quiescent, so keep the waiter and its framework
                    # pages alive. If the core recovers, a later poll (ours or the
                    # source pump's) will still deliver the terminal result.
                    if not poll_faulted:
                        self._note("op_poll_fault_waiting_terminal")
                        logger.warning(
                            "KVCRStore: polling op %s failed after submission; "
                            "retaining its host pages until the core reports "
                            "terminal.",
                            op_handle,
                            exc_info=True,
                        )
                        poll_faulted = True
                with self._poll_lock:
                    stashed = self._completed_ops.pop(op_handle, None)
                if stashed is not None:
                    self._note_entry_statuses(stashed)
                    return {k: v.success for k, v in stashed.items()}
                if not overdue and time.monotonic() >= deadline:
                    self._note("op_overdue_waiting_terminal")
                    logger.warning(
                        "KVCRStore: op %s did not complete within %.1fs; "
                        "retaining its host pages until the core reports "
                        "terminal.",
                        op_handle,
                        timeout,
                    )
                    overdue = True
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, _DRAIN_POLL_MAX_S)
        finally:
            with self._poll_lock:
                self._waiting_ops.discard(op_handle)
                # A completion can land between the last poll and here; drop it
                # now rather than leave it for a pop that will never come.
                self._completed_ops.pop(op_handle, None)

    # ------------------------------------------------------------------
    # v1 zero-copy interface (HiRadixCache path)
    # ------------------------------------------------------------------
    #
    # HiCache has two zero-copy call shapes and a backend needs both: the
    # HybridCacheController drives `batch_*_v2` with PoolTransfers, while
    # HiRadixCache's `_page_{get,set}_zero_copy` drives `batch_*_v1` with
    # (keys, host_indices). Only the pool name differs, so v1 wraps v2.

    def _kv_transfer(self, keys: List[str], host_indices) -> PoolTransfer:
        return PoolTransfer(
            name=PoolName.KV, host_indices=host_indices, keys=list(keys)
        )

    # v1 delegates to an already-guarded v2, so the guard here only covers what
    # v1 itself does -- building the PoolTransfer and reading the result out.
    @_fail_closed(_miss_per_key)
    def batch_set_v1(
        self,
        keys: List[str],
        host_indices,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self._logical_anchor:
            return [True] * len(keys)
        results = self.batch_set_v2([self._kv_transfer(keys, host_indices)], extra_info)
        return results.get(str(PoolName.KV), [False] * len(keys))

    @_fail_closed(_miss_per_key)
    def batch_get_v1(
        self,
        keys: List[str],
        host_indices,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self._logical_anchor:
            return [True] * len(keys)
        results = self.batch_get_v2([self._kv_transfer(keys, host_indices)], extra_info)
        return results.get(str(PoolName.KV), [False] * len(keys))

    @_fail_closed(lambda self, *a, **kw: 0)
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if self._logical_anchor:
            return len(keys)
        return self.batch_exists_v2(keys, None, extra_info).kv_hit_pages

    def clear(self) -> None:
        """Not supported: KVCR exposes no bulk invalidation.

        ``HiCacheStorage.clear`` is a bare ``pass``, so inheriting it makes
        ``/flush_cache`` report success while every block this worker deposited
        stays resident and peer-visible. That is worse than an error: the
        operator's reason for flushing -- a poisoned tier, a model swap -- is
        exactly the case where a stale block being served to a peer is a
        correctness bug, and the caller (``clear_storage_backend``) has a False
        return that says "this backend cannot".

        The core has ``release()`` for handles this store holds and eviction
        driven by capacity pressure, but nothing that drops a block by key, and
        the tier's contents are not enumerable from here. Implementing this
        needs a KVCR-side invalidate; until then it refuses honestly.
        """
        self._note("clear_unsupported")
        raise NotImplementedError(
            "KVCRStore does not support clear(): the KVCR core has no bulk "
            "invalidation, so blocks already deposited would stay resident and "
            "peer-visible after a flush that reported success."
        )

    # ------------------------------------------------------------------
    # byte-copy legacy ABC methods -- draft stubs
    # ------------------------------------------------------------------

    def get(self, key, target_location=None, target_sizes=None):
        return None  # DRAFT-STUB: v2 path is the supported one.

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)  # DRAFT-STUB

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        return False  # DRAFT-STUB

    def batch_set(
        self, keys, values=None, target_locations=None, target_sizes=None
    ) -> bool:
        return False  # DRAFT-STUB

    @_fail_closed(lambda self, *a, **kw: False)
    def exists(self, key: str) -> bool:
        if self._kvcr is None or self._segments_per_page is None:
            return False
        return self._locally_resident(self._page_segment_keys(key))
