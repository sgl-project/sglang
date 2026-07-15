from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import torch.distributed as dist
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
    KVTransferMetric,
    StateType,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)


class KVTransferError(Exception):
    def __init__(
        self,
        bootstrap_room: int,
        failure_reason: str,
        is_from_another_rank: bool = False,
    ):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason
        self.is_from_another_rank = is_from_another_rank

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


@dataclasses.dataclass
class PrefillServerInfo:
    # Topology fields (fetched from bootstrap server)
    attn_tp_size: int
    attn_cp_size: int
    dp_size: int
    pp_size: int
    page_size: Optional[int]
    kv_cache_dtype: Optional[str]
    follow_bootstrap_room: bool
    enable_dsa_cache_layer_split: bool = False

    # PD true-retraction rebootstrap: the prefill's HTTP API port. The decode
    # already knows the prefill host (the bootstrap_addr host), so it can POST
    # /generate to http://{bootstrap_host}:{prefill_http_port} to trigger a KV
    # recompute -- no router-injected pd_rebootstrap_prefill_url needed.
    prefill_http_port: Optional[int] = None

    # Pre-computed rank mapping (set by try_ensure_parallel_info on decode side)
    target_tp_rank: Optional[int] = None
    target_tp_ranks: Optional[List[int]] = None
    target_cp_ranks: Optional[List[int]] = None
    target_pp_ranks: Optional[List[int]] = None
    required_dst_info_num: Optional[int] = None
    required_prefill_response_num: Optional[int] = None

    def __post_init__(self):
        self.attn_tp_size = int(self.attn_tp_size)
        self.attn_cp_size = int(self.attn_cp_size)
        self.dp_size = int(self.dp_size)
        self.pp_size = int(self.pp_size)
        self.page_size = int(self.page_size) if self.page_size is not None else None
        self.kv_cache_dtype = (
            str(self.kv_cache_dtype) if self.kv_cache_dtype is not None else None
        )
        self.follow_bootstrap_room = bool(self.follow_bootstrap_room)
        self.enable_dsa_cache_layer_split = bool(self.enable_dsa_cache_layer_split)
        self.prefill_http_port = (
            int(self.prefill_http_port) if self.prefill_http_port is not None else None
        )


@dataclasses.dataclass
class PrefillRankInfo:
    rank_ip: str
    rank_port: int

    def __post_init__(self):
        self.rank_ip = str(self.rank_ip)
        self.rank_port = int(self.rank_port)


class CommonKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        self.kv_args = args
        self.kv_item_lens_sum = sum(args.kv_item_lens)
        self.state_item_lens_sum = sum(x for comp in args.state_item_lens for x in comp)
        self.is_mla_backend = is_mla_backend
        self.is_hybrid_mla_backend = getattr(args, "is_hybrid_mla_backend", False)
        self.disaggregation_mode = disaggregation_mode
        self.server_args = server_args
        # for p/d multi node infer
        self.bootstrap_host = server_args.host
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_cp_size = get_parallel().attn_cp_size
        self.attn_cp_rank = get_parallel().attn_cp_rank
        self.attn_dp_size = get_attention_dp_size()
        self.attn_dp_rank = get_attention_dp_rank()
        self.system_dp_size = (
            1 if server_args.enable_dp_attention else server_args.dp_size
        )
        self.system_dp_rank = (
            self.kv_args.system_dp_rank if self.kv_args.system_dp_rank else 0
        )
        self.pp_size = server_args.pp_size
        self.pp_rank = self.kv_args.pp_rank
        self.local_ip = get_local_ip_auto()
        cp_sharded_prefill = self.attn_cp_size > 1 and (
            self.is_hybrid_mla_backend or server_args.enable_dsa_cache_layer_split
        )

        hybrid_decode_pulls_all_ranks = (
            self.is_hybrid_mla_backend
            and disaggregation_mode == DisaggregationMode.DECODE
        )
        self.enable_all_cp_ranks_for_transfer = (
            envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get()
            or cp_sharded_prefill
            or hybrid_decode_pulls_all_ranks
        )

        # bind zmq socket
        self._zmq_ctx = zmq.Context()
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            self._zmq_ctx, zmq.PULL, host=self.local_ip
        )
        logger.debug(f"kv manager bind to {self.local_ip}:{self.rank_port}")

        self.request_status: Dict[int, KVPoll] = {}
        self._socket_cache: Dict[str, zmq.Socket] = {}
        self._monitor_cache: Dict[str, zmq.Socket] = {}
        self._socket_lock = threading.Lock()
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # When SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER is True, all CP ranks
            # participate in KV transfer; Otherwise only CP rank 0 sends.
            self.is_dummy_cp_rank = (
                not self.enable_all_cp_ranks_for_transfer
                and self.attn_cp_size > 1
                and self.attn_cp_rank != 0
            )
            # Sync the leader's bootstrap port to every rank before
            # registering: in multi-node prefill, registration targets
            # `dist_init_addr` (rank 0) but each rank's local port may
            # differ when the launcher auto-reserves a free port per host.
            self.bootstrap_port = self._sync_bootstrap_port_across_nodes(
                self.bootstrap_port
            )
            self.register_to_bootstrap()
            self.transfer_infos = {}
            self.req_to_decode_prefix_len: Dict[int, int] = {}
            self.req_to_dspark_hidden_meta: Dict[int, dict] = {}
            self.decode_kv_args_table = {}
            self.pp_group = get_pp_group()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            self.bootstrap_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.enable_staging: bool = False
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.connection_lock = threading.Lock()
            self.required_prefill_response_num_table: Dict[int, int] = {}
            self.prefill_info_table: Dict[str, PrefillServerInfo] = {}
            self.heartbeat_failures: Dict[str, int] = {}
            self.session_pool: Dict = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker: Dict[str, Set[int]] = defaultdict(set)
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
            )
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
            # PD true-retraction rebootstrap: a shared executor + per-thread HTTP
            # sessions used to drive the original prefill worker's ``/generate``
            # endpoint so it recomputes a retracted request's prefix KV under the
            # current weights. Created lazily on first use so deployments that
            # never retract pay nothing.
            self._prefill_recompute_executor: Optional[
                concurrent.futures.ThreadPoolExecutor
            ] = None
            self._prefill_recompute_executor_lock = threading.Lock()
            self._prefill_recompute_sessions = threading.local()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def check_status(self, bootstrap_room: int) -> KVPoll:
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            # Do not resurrect a cleared entry with Failed: once clear() has
            # popped the room from request_status, any late update_status(Failed)
            # (e.g. from abort()) must be a no-op. Otherwise a Failed entry could
            # pollute a future request that reuses the same bootstrap_room.
            if status == KVPoll.Failed:
                return
            self.request_status[bootstrap_room] = status
        else:
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def _ensure_prefill_recompute_executor(
        self,
    ) -> concurrent.futures.ThreadPoolExecutor:
        """Lazily create the shared executor that drives PD-retract rebootstrap
        ``/generate`` calls. One executor per (decode) kv manager, shared across
        all receivers."""
        executor = self._prefill_recompute_executor
        if executor is not None:
            return executor
        with self._prefill_recompute_executor_lock:
            if self._prefill_recompute_executor is None:
                workers = envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
                if workers is None:
                    workers = 16
                self._prefill_recompute_executor = (
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=max(1, workers),
                        thread_name_prefix="pd-rebootstrap-prefill",
                    )
                )
            return self._prefill_recompute_executor

    def _get_prefill_recompute_session(self) -> requests.Session:
        """Per-thread ``requests.Session`` for the rebootstrap executor threads
        (``requests.Session`` is not safe for concurrent cross-thread use)."""
        session = getattr(self._prefill_recompute_sessions, "session", None)
        if session is None:
            session = requests.Session()
            self._prefill_recompute_sessions.session = session
        return session

    def _resolve_rebootstrap_prefill_url(
        self, kv_receiver: CommonKVReceiver
    ) -> Optional[str]:
        """Derive the prefill ``/generate`` base URL for a PD true-retraction
        rebootstrap from bootstrap info.

        The decode already knows the prefill host (the ``bootstrap_addr`` host,
        which the router/client set to the prefill's HTTP host), and the prefill
        self-registers its HTTP API port in ``PrefillServerInfo`` at bootstrap
        registration. Combining them yields ``http://{host}:{prefill_http_port}``
        with no router-injected ``pd_rebootstrap_prefill_url``.
        """
        prefill_info = self.prefill_info_table.get(kv_receiver.bootstrap_addr)
        if prefill_info is None or prefill_info.prefill_http_port is None:
            return None
        host = NetworkAddress.parse(kv_receiver.bootstrap_addr).host
        return NetworkAddress(host, prefill_info.prefill_http_port).to_url()

    def submit_prefill_recompute(
        self, kv_receiver: CommonKVReceiver, payload: dict
    ) -> None:
        """Dispatch a PD true-retraction rebootstrap ``/generate`` to the
        original prefill worker so it recomputes the retracted request's prefix
        KV under the current weights and transfers it back over the
        already-bootstrapped channel.

        The target prefill ``/generate`` URL is derived from bootstrap info (the
        prefill self-registered its HTTP port), not from a router-injected field.

        Non-blocking from the scheduler's perspective: the HTTP POST runs on the
        shared executor. Any failure (unresolved URL, HTTP error, exception) is
        surfaced through the standard ``KVPoll.Failed`` path via
        ``kv_receiver.abort()`` so the scheduler's existing transfer-failure
        handling streams the aborted request back to the client. ``payload`` is
        prebuilt by the decode scheduler (``Req.build_rebootstrap_payload``) so
        HTTP/sampling concerns stay on the kv manager.

        The decode scheduler broadcasts each retracted request to every rank in
        its attention TP/CP group and every PP stage, so all of them reach this
        call and would each POST an identical ``/generate`` -- making the prefill
        worker recompute the same request once per decode rank. The ``/generate``
        is a server-level call: the prefill frontend fans it out to its own
        workers and transfers the recomputed KV back to *all* decode ranks, so
        exactly one decode rank must issue it. Elect the same leader the request
        receiver uses (attn-tp/attn-cp group leader, first PP stage); the other
        ranks still bootstrap and receive their KV shard as usual, and on failure
        the leader-only abort matches the leader-only output streaming (other
        ranks fall back to the per-request waiting-timeout safety net).
        """
        if self.attn_tp_rank != 0 or self.attn_cp_rank != 0 or self.pp_rank != 0:
            return
        prefill_url = self._resolve_rebootstrap_prefill_url(kv_receiver)
        if not prefill_url:
            logger.error(
                "PD retract rebootstrap could not resolve the prefill /generate "
                "URL from bootstrap info (rid=%s bootstrap_room=%s bootstrap_addr=%s).",
                payload.get("rid"),
                payload.get("bootstrap_room"),
                kv_receiver.bootstrap_addr,
            )
            self._fail_prefill_recompute(
                kv_receiver,
                "PD retract rebootstrap could not resolve the prefill /generate "
                "URL from bootstrap info.",
            )
            return
        self._ensure_prefill_recompute_executor().submit(
            self._run_prefill_recompute, kv_receiver, prefill_url, payload
        )

    def _fail_prefill_recompute(
        self, kv_receiver: CommonKVReceiver, reason: str
    ) -> None:
        """Fail a rebootstrap request via the standard ``KVPoll.Failed`` path.

        ``abort()`` transitions the receiver to Failed and notifies the prefill
        worker to release its orphaned bootstrap entry, but records a generic
        reason; we overwrite it with a descriptive one so the eventual
        ``failure_exception`` (and the client-facing abort message) explains that
        the rebootstrap ``/generate`` failed rather than reporting a spurious
        ``AbortReq``.
        """
        kv_receiver.abort()
        self.record_failure(kv_receiver.bootstrap_room, reason)

    def _run_prefill_recompute(
        self, kv_receiver: CommonKVReceiver, prefill_url: str, payload: dict
    ) -> None:
        rid = payload.get("rid")
        try:
            response = self._get_prefill_recompute_session().post(
                prefill_url.rstrip("/") + "/generate",
                json=payload,
                timeout=self.waiting_timeout,
            )
            if response.status_code >= 400:
                logger.error(
                    "PD rebootstrap prefill failed for rid=%s status=%s body=%s",
                    rid,
                    response.status_code,
                    response.text[:512],
                )
                self._fail_prefill_recompute(
                    kv_receiver,
                    f"PD retract rebootstrap /generate failed for rid={rid} "
                    f"(status={response.status_code}).",
                )
        except Exception:
            logger.exception("PD rebootstrap prefill request failed for rid=%s", rid)
            self._fail_prefill_recompute(
                kv_receiver,
                f"PD retract rebootstrap /generate request errored for rid={rid}.",
            )

    def try_ensure_parallel_info(self, bootstrap_addr: str) -> bool:
        """Single non-blocking attempt to fetch and cache prefill parallel info.
        Returns True if info is available (cached or freshly fetched)."""
        if bootstrap_addr in self.prefill_info_table:
            return True

        info: PrefillServerInfo = None
        try:
            url = f"http://{bootstrap_addr}/route?prefill_dp_rank={-1}&prefill_cp_rank={-1}&target_tp_rank={-1}&target_pp_rank={-1}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                info = PrefillServerInfo(**data)
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Error fetching prefill server info from bootstrap: {e}")
            return False

        # Sanity checks
        if info.page_size is not None and info.page_size != self.kv_args.page_size:
            raise RuntimeError(
                f"Page size mismatch: prefill server has page_size={info.page_size}, "
                f"but decode server has page_size={self.kv_args.page_size}. "
                f"Both servers must use the same --page-size value."
            )

        if (
            info.kv_cache_dtype is not None
            and info.kv_cache_dtype != self.server_args.kv_cache_dtype
        ):
            raise RuntimeError(
                f"KV cache dtype mismatch: prefill server has kv_cache_dtype={info.kv_cache_dtype}, "
                f"but decode server has kv_cache_dtype={self.server_args.kv_cache_dtype}. "
                f"Both servers must use the same --kv-cache-dtype value."
            )

        self._resolve_rank_mapping(info)
        self.prefill_info_table[bootstrap_addr] = info
        logger.debug(f"Prefill parallel info for [{bootstrap_addr}]: {info}")
        return True

    def _resolve_rank_mapping(self, info: PrefillServerInfo) -> None:
        """Compute TP/CP/PP rank mapping and store on the PrefillServerInfo object.
        Deterministic for a given (bootstrap_addr, decode engine) pair."""
        # TP rank mapping
        if self.attn_tp_size == info.attn_tp_size:
            target_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
            required_dst_info_num = 1
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        elif self.attn_tp_size > info.attn_tp_size:
            if not self.is_mla_backend and not self.is_hybrid_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            target_tp_rank = (self.kv_args.engine_rank % self.attn_tp_size) // (
                self.attn_tp_size // info.attn_tp_size
            )
            required_dst_info_num = self.attn_tp_size // info.attn_tp_size
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        else:
            if not self.is_mla_backend and not self.is_hybrid_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks
            target_tp_ranks = list(
                range(
                    (self.kv_args.engine_rank % self.attn_tp_size)
                    * (info.attn_tp_size // self.attn_tp_size),
                    (self.kv_args.engine_rank % self.attn_tp_size + 1)
                    * (info.attn_tp_size // self.attn_tp_size),
                )
            )
            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            target_tp_rank = target_tp_ranks[0]
            required_dst_info_num = 1
            if self.is_mla_backend:
                required_prefill_response_num = 1
            else:
                required_prefill_response_num = info.attn_tp_size // self.attn_tp_size

        # CP rank mapping — decode cp size should be equal to 1
        assert self.attn_cp_size == 1, (
            f"Decode cp size ({self.attn_cp_size}) should be equal to 1",
        )
        if self.attn_cp_size == info.attn_cp_size:
            assert info.attn_cp_size == 1, (
                f"When prefill cp size is 1, attn cp size should be 1, but got {self.attn_cp_size}",
            )
            target_cp_ranks = [self.attn_cp_rank]
        else:
            target_cp_ranks = list(range(info.attn_cp_size))
            pull_from_all_cp_ranks = (
                self.enable_all_cp_ranks_for_transfer
                or info.enable_dsa_cache_layer_split
            )
            if not pull_from_all_cp_ranks:
                # Only retrieve from prefill CP rank 0 when not using all ranks
                target_cp_ranks = target_cp_ranks[:1]
                required_prefill_response_num *= 1
            else:
                required_prefill_response_num *= info.attn_cp_size // self.attn_cp_size

        # PP rank mapping — decode pp size should be equal to prefill pp size or 1
        assert self.pp_size == info.pp_size or self.pp_size == 1, (
            f"Decode pp size ({self.pp_size}) should be equal to prefill pp size ({info.pp_size}) or 1",
        )
        if info.pp_size == self.pp_size:
            target_pp_ranks = [self.pp_rank]
        else:
            target_pp_ranks = list(range(info.pp_size))
            required_prefill_response_num *= info.pp_size // self.pp_size

        info.target_tp_rank = target_tp_rank
        info.target_tp_ranks = target_tp_ranks
        info.target_cp_ranks = target_cp_ranks
        info.target_pp_ranks = target_pp_ranks
        info.required_dst_info_num = required_dst_info_num
        info.required_prefill_response_num = required_prefill_response_num

    def _sync_bootstrap_port_across_nodes(self, local_port: int) -> int:
        """Broadcast world-rank-0's bootstrap port to all prefill ranks.

        Required for multi-node prefill when the launcher auto-reserves a
        free port per host (e.g. Dynamo's
        `_reserve_disaggregation_bootstrap_port`): without sync, non-leader
        ranks register to `<leader_ip>:<their_local_port>`, hit
        `Connection refused`, and the leader's `prefill_port_table` ends
        up missing rows.
        """
        if not self.dist_init_addr or self.server_args.nnodes == 1:
            return local_port

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "torch.distributed must be initialised before "
                "CommonKVManager registers to the bootstrap server in "
                "multi-node prefill mode."
            )

        world_group = get_world_group()
        synced_port = world_group.broadcast_object(local_port, src=0)
        if synced_port != local_port:
            logger.info(
                f"Synced disaggregation bootstrap port from leader: "
                f"local={local_port} -> leader={synced_port} "
                f"(world_rank={world_group.rank_in_group})"
            )
        return synced_port

    def register_to_bootstrap(self):
        """Register prefill server info to bootstrap server via HTTP PUT."""
        if self.dist_init_addr:
            # Multi-node case: bootstrap server's host is dist_init_addr
            host = NetworkAddress.parse(self.dist_init_addr).resolved().host
        else:
            # Single-node case: bootstrap server's host is the same as http server's host
            host = self.bootstrap_host
            # A wildcard bind address (0.0.0.0 / ::) is not a valid HTTP Host
            # and can't be connected to; rewrite it to the same-family loopback,
            # which the wildcard listener also binds.  (self.local_ip is wrong
            # here — it can resolve to a different family than the listener,
            # e.g. IPv6 while the server is bound to 0.0.0.0.)
            host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)

        bootstrap_na = NetworkAddress(host, self.bootstrap_port)
        url = f"{bootstrap_na.to_url()}/route"
        payload = {
            "attn_tp_size": self.attn_tp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_cp_size": self.attn_cp_size,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_dp_size": self.attn_dp_size,
            "attn_dp_rank": self.attn_dp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "system_dp_size": self.system_dp_size,
            "system_dp_rank": self.system_dp_rank,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            "page_size": self.kv_args.page_size,
            "kv_cache_dtype": self.server_args.kv_cache_dtype,
            "load_balance_method": self.server_args.load_balance_method,
            "enable_dsa_cache_layer_split": getattr(
                self.server_args, "enable_dsa_cache_layer_split", False
            ),
            # Self-register the HTTP API port so the decode can derive the PD
            # retract rebootstrap /generate URL from bootstrap info instead of a
            # router-injected pd_rebootstrap_prefill_url.
            "prefill_http_port": self.server_args.port,
        }

        max_retries, initial_delay, max_delay = 5, 1.0, 30.0
        for attempt in range(max_retries):
            try:
                response = requests.put(url, json=payload, timeout=5)
                if response.status_code == 200:
                    logger.debug("Prefill successfully registered to bootstrap server.")
                    return
                logger.warning(
                    f"Prefill register attempt {attempt + 1}/{max_retries} failed: status {response.status_code}"
                )
            except Exception as e:
                # Walk to root cause to skip misleading urllib3 wrapper messages
                cause = e
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                logger.warning(
                    f"Prefill register attempt {attempt + 1}/{max_retries} failed: {cause}"
                )
            if attempt == max_retries - 1:
                break
            delay = min(initial_delay * (2**attempt), max_delay) * (
                0.75 + 0.25 * (time.monotonic() % 1)
            )
            time.sleep(delay)
        logger.error(
            f"Prefill instance failed to register to bootstrap server after {max_retries} retries"
        )

    def _connect(self, endpoint: str, is_ipv6: bool = False):
        with self._socket_lock:
            sock = self._socket_cache.get(endpoint)
            if sock is not None:
                monitor = self._monitor_cache.get(endpoint)
                disconnected = False
                if monitor is not None:
                    try:
                        monitor.recv_multipart(zmq.NOBLOCK)
                        disconnected = True
                    except zmq.Again:
                        pass
                    except zmq.ZMQError:
                        disconnected = True
                if not disconnected:
                    return sock
                sock.close(linger=0)
                if monitor is not None:
                    monitor.close()
                self._socket_cache.pop(endpoint, None)
                self._monitor_cache.pop(endpoint, None)

            sock = self._zmq_ctx.socket(zmq.PUSH)
            if is_ipv6:
                sock.setsockopt(zmq.IPV6, 1)
            sock.setsockopt(zmq.RECONNECT_IVL, -1)
            sock.setsockopt(zmq.SNDTIMEO, 30000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
            sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
            sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
            sock.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)
            sock.connect(endpoint)
            self._socket_cache[endpoint] = sock
            self._monitor_cache[endpoint] = sock.get_monitor_socket(
                zmq.EVENT_DISCONNECTED
            )
            return sock

    def get_mha_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int], int]:
        start_layer = self.kv_args.prefill_start_layer
        num_kv_layers = len(src_kv_ptrs) // 2
        end_layer = start_layer + num_kv_layers
        dst_num_total_layers = len(dst_kv_ptrs) // 2
        src_k_ptrs = src_kv_ptrs[:num_kv_layers]
        src_v_ptrs = src_kv_ptrs[num_kv_layers:]
        if num_kv_layers == dst_num_total_layers:
            dst_k_ptrs = dst_kv_ptrs[:dst_num_total_layers]
            dst_v_ptrs = dst_kv_ptrs[dst_num_total_layers:]
        elif (
            num_kv_layers < dst_num_total_layers
            and dst_num_total_layers % num_kv_layers != 0
        ):
            # Case: Decode has draft model KV while Prefill is deployed without speculative decoding
            # dst_kv_ptrs layout: [K_main..., V_main..., draft_K..., draft_V...]
            multiplier_ratio = dst_num_total_layers // num_kv_layers
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            v_ptr_offset = num_kv_layers * multiplier_ratio
            dst_v_ptrs = dst_kv_ptrs[
                v_ptr_offset + start_layer : v_ptr_offset + end_layer
            ]
        else:
            # Decode pp size should be equal to prefill pp size or 1
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            dst_v_ptrs = dst_kv_ptrs[
                dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
            ]
        layers_current_pp_stage = len(src_k_ptrs)
        return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage

    def get_mla_kv_ptrs_with_pp(
        self,
        src_kv_ptrs: List[int],
        dst_kv_ptrs: List[int],
        state_type: Optional[StateType] = None,
    ) -> Tuple[List[int], List[int], int]:
        # Fast path: both sides use exactly the same PP layout
        if len(src_kv_ptrs) == len(dst_kv_ptrs):
            return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)

        mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
        if mla_ratios:
            # Compressed-MLA (e.g. DeepSeek V4): the flat list is organized
            # by buffer type (compression-ratio bucket) rather than by
            # layer, so we locate the sub-range for this PP stage inside each
            # section of the dst flat list.
            sliced_src_kv_ptrs, sliced_dst_kv_ptrs = self._mla_slice_ptrs_for_pp(
                src_kv_ptrs, dst_kv_ptrs, mla_ratios, state_type
            )
            return (
                sliced_src_kv_ptrs,
                sliced_dst_kv_ptrs,
                len(sliced_src_kv_ptrs),
            )

        # Regular MLA PP slicing
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + len(src_kv_ptrs)
        # Decode pp size should be equal to prefill pp size or 1
        sliced_dst_kv_ptrs = dst_kv_ptrs[start_layer:end_layer]
        return src_kv_ptrs, sliced_dst_kv_ptrs, len(src_kv_ptrs)

    def _mla_slice_ptrs_for_pp(
        self,
        src_kv_ptrs: List[int],
        dst_kv_ptrs: List[int],
        mla_ratios: List[int],
        state_type: Optional[StateType] = None,
    ) -> Tuple[List[int], List[int]]:
        """Produce aligned (src, dst) pointer lists for compressed-MLA
        pools (e.g. DeepSeek V4) under PP.

        The pool produces two possible flat-list layouts (selected via dst
        length):

        - kv_data layout, length = 2 * c4_L + c128_L:
            [c4_layer_{0..c4_L-1},
             c4_indexer_layer_{0..c4_L-1},
             c128_layer_{0..c128_L-1}]
          Each section is indexed by compressed-layer id within that
          compression bucket.

        - SWA state_data layout, length = swa_L + 2 * c4_L:
            [swa_layer_{0..swa_L-1},
             c4_compress_state_{0..c4_L-1},
             c4_indexer_compress_state_{0..c4_L-1}]
          ``swa_L`` is the SWA pool's actual buffer count
          (``num_effective_layers``), which can be smaller than
          ``len(mla_ratios)`` when the HF config's ``compress_ratios``
          list contains entries for layers not materialized into the SWA
          pool (e.g. an MTP/nextn slot at the tail).

        - C128_STATE layout, length = c128_L:
            [c128_compress_state_{0..c128_L-1}]

        src is already PP-filtered on the prefill side. dst is the
        decode-side full-model list (when decode is PP=1). We slice dst to
        match src's PP stage. If src itself is also full-model, it is
        returned unchanged.
        """
        start_layer = self.kv_args.prefill_start_layer
        end_layer = getattr(self.kv_args, "prefill_end_layer", None)
        assert (
            end_layer is not None
        ), "KVArgs.prefill_end_layer must be set when using compressed-MLA PD with PP"

        c4_full = sum(1 for r in mla_ratios if r == 4)
        c128_full = sum(1 for r in mla_ratios if r == 128)
        kv_layout_len = 2 * c4_full + c128_full

        c4_off_s = sum(1 for r in mla_ratios[:start_layer] if r == 4)
        c4_off_e = sum(1 for r in mla_ratios[:end_layer] if r == 4)
        c128_off_s = sum(1 for r in mla_ratios[:start_layer] if r == 128)
        c128_off_e = sum(1 for r in mla_ratios[:end_layer] if r == 128)

        if state_type == StateType.C128_STATE:
            return src_kv_ptrs, list(dst_kv_ptrs[c128_off_s:c128_off_e])

        if state_type == StateType.SWA_RING:
            swa_s = min(start_layer, len(dst_kv_ptrs))
            swa_e = min(end_layer, len(dst_kv_ptrs))
            return src_kv_ptrs, list(dst_kv_ptrs[swa_s:swa_e])

        if (
            state_type not in (StateType.SWA, StateType.SWA_RING, StateType.C128_STATE)
            and len(dst_kv_ptrs) == kv_layout_len
        ):
            sliced_dst = (
                list(dst_kv_ptrs[c4_off_s:c4_off_e])
                + list(dst_kv_ptrs[c4_full + c4_off_s : c4_full + c4_off_e])
                + list(dst_kv_ptrs[2 * c4_full + c128_off_s : 2 * c4_full + c128_off_e])
            )
            return src_kv_ptrs, sliced_dst

        # SWA state-data layout. ``swa_L`` is derived from the actual dst
        # length so we tolerate cases where the SWA pool has fewer buffers
        # than ``len(mla_ratios)`` (e.g. nextn padding). C128 state ships as
        # a separate StateType.C128_STATE component and must not be counted
        # here.
        swa_L = len(dst_kv_ptrs) - 2 * c4_full
        if swa_L < 0 or swa_L > len(mla_ratios):
            raise ValueError(
                f"Unexpected compressed-MLA dst_kv_ptrs length "
                f"{len(dst_kv_ptrs)}; expected either {kv_layout_len} "
                f"(kv_data) or swa_L + {2 * c4_full} "
                f"(state_data) given compression_ratios "
                f"(c4={c4_full}, c128={c128_full}, "
                f"total={len(mla_ratios)})."
            )

        swa_s = min(start_layer, swa_L)
        swa_e = min(end_layer, swa_L)
        compress_section_start = swa_L
        indexer_section_start = swa_L + c4_full
        sliced_dst = (
            list(dst_kv_ptrs[swa_s:swa_e])
            + list(
                dst_kv_ptrs[
                    compress_section_start
                    + c4_off_s : compress_section_start
                    + c4_off_e
                ]
            )
            + list(
                dst_kv_ptrs[
                    indexer_section_start + c4_off_s : indexer_section_start + c4_off_e
                ]
            )
        )

        return src_kv_ptrs, sliced_dst

    def _start_heartbeat_checker_thread(self):
        """Start the heartbeat checker thread for Decode worker."""

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_info_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0
                            self._on_heartbeat_success(bootstrap_addr)
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=heartbeat_checker, daemon=True).start()

    def _on_heartbeat_success(self, bootstrap_addr: str):
        """Hook called on successful heartbeat. Override for backend-specific cleanup."""
        pass

    def _handle_node_failure(self, failed_bootstrap_addr: str):
        """Handle failure of a prefill node."""
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            # Collect TCP endpoints from cached bootstrap_infos before deletion
            stale_endpoints = set()
            for k in keys_to_remove:
                for info in self.connection_pool[k]:
                    ip = info.get("rank_ip")
                    port = info.get("rank_port")
                    if ip and port:
                        na = NetworkAddress(ip, int(port))
                        stale_endpoints.add(na.to_tcp())
            for k in keys_to_remove:
                del self.connection_pool[k]
            self.prefill_info_table.pop(failed_bootstrap_addr, None)

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            self.addr_to_rooms_tracker.pop(failed_bootstrap_addr, None)

        for endpoint in stale_endpoints:
            CommonKVReceiver.disconnect_endpoint(endpoint)

        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)

        logger.error(
            f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), "
            f"{len(affected_rooms)} requests affected"
        )


class CommonKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state: Optional[KVPoll] = None
        self._transfer_metric = KVTransferMetric()
        self._transfer_num_kv_indices = 0
        self._transfer_num_state_indices = 0
        # inner state
        self.curr_idx = 0
        self.init_time: Optional[float] = None
        if self.kv_mgr.is_dummy_cp_rank:
            # Non-authoritative CP ranks are dummy participants.
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        if self.kv_mgr.server_args.dp_size > 1 and not req_has_disagg_prefill_dp_rank:
            if self.kv_mgr.server_args.load_balance_method != "follow_bootstrap_room":
                self._register_prefill_dp_rank()
            elif (
                self.kv_mgr.attn_dp_rank
                != self.bootstrap_room % self.kv_mgr.server_args.dp_size
            ):
                # follow_bootstrap_room was overridden by external routed_dp_rank
                if envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get():
                    self._register_prefill_dp_rank()
                else:
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"follow_bootstrap_room conflict: dispatched to dp_rank "
                        f"{self.kv_mgr.attn_dp_rank} but bootstrap_room "
                        f"{self.bootstrap_room} implies dp_rank "
                        f"{self.bootstrap_room % self.kv_mgr.server_args.dp_size}. "
                        f"Set SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1 "
                        f"to allow mixed routing.",
                    )
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

    def _register_prefill_dp_rank(self):
        """Register this request's prefill dp_rank to the bootstrap server."""
        url = f"http://{self.bootstrap_server_url}/register_dp_rank"
        payload = {
            "bootstrap_room": self.bootstrap_room,
            "dp_rank": self.kv_mgr.attn_dp_rank,
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                logger.error(
                    f"Failed to register prefill dp_rank: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(f"Failed to register prefill dp_rank: {e}")

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        logger.debug(
            f"CommonKVSender init with num_kv_indices: {num_kv_indices} and aux_index: {aux_index}"
        )

    def pop_decode_prefix_len(self) -> int:
        return self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, 0)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0 or last_chunk

    def get_transfer_metric(self) -> KVTransferMetric:
        total_bytes = self._transfer_num_kv_indices * self.kv_mgr.kv_item_lens_sum
        total_bytes += (
            self._transfer_num_state_indices * self.kv_mgr.state_item_lens_sum
        )
        self._transfer_metric.transfer_total_bytes = total_bytes
        return self._transfer_metric

    def _record_transfer_indices(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List],
    ):
        self._transfer_num_kv_indices += len(kv_indices)
        if state_indices:
            for component_indices in state_indices:
                if component_indices is not None:
                    self._transfer_num_state_indices += len(component_indices)

    def _prepare_send_indices(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ) -> Tuple[npt.NDArray[np.int32], slice, bool, bool]:
        """Common pre-processing for send(): index tracking and CP-rank handling.

        Returns:
            (kv_indices, index_slice, is_last_chunk, should_skip)
            If should_skip is True, the caller should return immediately.
        """
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last_chunk = self.curr_idx == self.num_kv_indices

        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not self.kv_mgr.server_args.enable_dsa_cache_layer_split
        ):
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
                total_pages=self.num_kv_indices,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            if not is_last_chunk:
                return kv_indices, index_slice, is_last_chunk, True
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return kv_indices, index_slice, is_last_chunk, True

        return kv_indices, index_slice, is_last_chunk, False

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        pass

    def _check_bootstrap_timeout(self) -> Optional[KVPoll]:
        if self.init_time is None:
            return None
        elapsed = time.time() - self.init_time
        if elapsed < self.kv_mgr.bootstrap_timeout:
            return None
        logger.warning_once(
            "Some requests timed out when bootstrapping, "
            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
        )
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
            f"in KVPoll.Bootstrapping",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        return KVPoll.Failed

    def poll(self) -> KVPoll:
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "req_to_decode_prefix_len"):
            self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "req_to_dspark_hidden_meta"):
            self.kv_mgr.req_to_dspark_hidden_meta.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "transfer_infos"):
            self.kv_mgr.transfer_infos.pop(self.bootstrap_room, None)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed


class CommonKVReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.conclude_state: Optional[KVPoll] = None
        self.require_staging: bool = False
        self.init_time: Optional[float] = None
        self.abort_notified: bool = False
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

    def init(self, prefill_dp_rank: int):
        if self.bootstrap_addr not in self.kv_mgr.prefill_info_table:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Prefill server with bootstrap_addr: {self.bootstrap_addr} is healthy before, but now it is down. Request (bootstrap_room: {self.bootstrap_room}) has been marked as failed.",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Read pre-computed rank mapping from prefill_info (computed in try_ensure_parallel_info)
        self.prefill_info = self.kv_mgr.prefill_info_table[self.bootstrap_addr]
        self.target_tp_rank = self.prefill_info.target_tp_rank
        self.target_tp_ranks = self.prefill_info.target_tp_ranks
        self.target_cp_ranks = self.prefill_info.target_cp_ranks
        self.target_pp_ranks = self.prefill_info.target_pp_ranks
        self.required_dst_info_num = self.prefill_info.required_dst_info_num
        self.required_prefill_response_num = (
            self.prefill_info.required_prefill_response_num
        )

        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )

        if self.kv_mgr.enable_staging:
            self.require_staging = (
                self.prefill_info.attn_tp_size != 0
                and self.prefill_info.attn_tp_size != self.kv_mgr.attn_tp_size
            )

        self.prefill_dp_rank = prefill_dp_rank
        self._setup_bootstrap_infos()
        if self.conclude_state == KVPoll.Failed:
            return
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _setup_bootstrap_infos(self):
        all_bootstrap_infos = []
        # NOTE: key distinguished by bootstrap_addr, prefill_dp_rank, prefill_cp_rank, and target_tp_rank
        for target_cp_rank in self.target_cp_ranks:
            bootstrap_key = f"{self.bootstrap_addr}_{self.prefill_dp_rank}_{target_cp_rank}_{self.target_tp_rank}"

            if bootstrap_key not in self.kv_mgr.connection_pool:
                bootstrap_infos = []
                for target_tp_rank in self.target_tp_ranks:
                    # Enable higher PP ranks to be bootstrapped earlier to make PP PD requests bootstrap more robust
                    for target_pp_rank in reversed(self.target_pp_ranks):
                        bootstrap_info = self._get_bootstrap_info_from_server(
                            self.prefill_dp_rank,
                            target_cp_rank,
                            target_tp_rank,
                            target_pp_rank,
                        )
                        if bootstrap_info is not None:
                            if self.kv_mgr.is_mla_backend:
                                # For MLA: target_tp_rank is the selected real rank, others are dummy ranks
                                bootstrap_info["is_dummy"] = not bool(
                                    target_tp_rank == self.target_tp_rank
                                    or self.target_tp_rank is None
                                )
                            else:
                                # For non-MLA: all target_tp_ranks are selected real ranks
                                bootstrap_info["is_dummy"] = False
                            logger.debug(
                                f"Fetched bootstrap info: {bootstrap_info} for DP {self.prefill_dp_rank} CP {target_cp_rank} TP {target_tp_rank} PP {target_pp_rank}"
                            )
                            bootstrap_infos.append(bootstrap_info)
                        else:
                            self.kv_mgr.record_failure(
                                self.bootstrap_room,
                                f"Could not fetch bootstrap info for: prefill_dp_rank: {self.prefill_dp_rank} prefill_cp_rank: {target_cp_rank} target_tp_rank: {target_tp_rank} and target_pp_rank {target_pp_rank}",
                            )
                            self.conclude_state = KVPoll.Failed
                            self.kv_mgr.update_status(
                                self.bootstrap_room, KVPoll.Failed
                            )
                            self.bootstrap_infos = None
                            return

                self.bootstrap_infos = bootstrap_infos
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

                # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
                self._register_kv_args()
            else:
                self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

            assert len(self.bootstrap_infos) > 0
            all_bootstrap_infos.extend(self.bootstrap_infos)

        self.bootstrap_infos = all_bootstrap_infos

    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?prefill_dp_rank={prefill_dp_rank}&prefill_cp_rank={prefill_cp_rank}&target_tp_rank={target_tp_rank}&target_pp_rank={target_pp_rank}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    @staticmethod
    def query_prefill_dp_ranks(
        bootstrap_addr: str, bootstrap_rooms: List[int]
    ) -> Dict[str, int]:
        """Batch query prefill dp_ranks for given bootstrap_rooms."""
        try:
            url = f"http://{bootstrap_addr}/query_dp_ranks"
            response = requests.post(
                url,
                json={"bootstrap_rooms": bootstrap_rooms},
                timeout=5,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Failed to query dp_ranks: {response.status_code}, {response.text}"
                )
                return {}
        except Exception as e:
            logger.error(f"Error querying dp_ranks from bootstrap: {e}")
            return {}

    @classmethod
    def _connect(cls, endpoint: str, is_ipv6: bool = False):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.setsockopt(zmq.RECONNECT_IVL, -1)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @classmethod
    def disconnect_endpoint(cls, endpoint: str):
        with cls._global_lock:
            sock = cls._socket_cache.pop(endpoint, None)
            lock = cls._socket_locks.pop(endpoint, None)
        if sock:
            if lock:
                with lock:
                    sock.close()
            else:
                sock.close()
            logger.debug(f"Disconnected stale ZMQ PUSH socket (receiver): {endpoint}")

    @classmethod
    def _connect_to_bootstrap_server(cls, bootstrap_info: dict):
        ip_address = bootstrap_info["rank_ip"]
        port = bootstrap_info["rank_port"]
        na = NetworkAddress(ip_address, port)
        sock, lock = cls._connect(na.to_tcp(), is_ipv6=na.is_ipv6)
        return sock, lock

    def _register_kv_args(self):
        pass

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
        decode_prefix_len: Optional[int] = None,
        spec_metadata: Optional[dict] = None,
    ):
        raise NotImplementedError

    def _check_waiting_timeout(self) -> Optional[KVPoll]:
        if self.init_time is None:
            return None
        elapsed = time.time() - self.init_time
        if elapsed < self.kv_mgr.waiting_timeout:
            return None
        logger.warning_once(
            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
        )
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
            f"in KVPoll.WaitingForInput",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        if (
            not self.abort_notified
            and hasattr(self, "bootstrap_infos")
            and self.bootstrap_infos is not None
        ):
            self._send_abort_notification()
            self.abort_notified = True
        return KVPoll.Failed

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room, None)
        self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room, None)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed
        if (
            not self.abort_notified
            and hasattr(self, "bootstrap_infos")
            and self.bootstrap_infos is not None
        ):
            self._send_abort_notification()
            self.abort_notified = True

    def _send_abort_notification(self):
        for bootstrap_info in self.bootstrap_infos:
            # Best-effort notification to prefill side that this request was aborted.
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            b"ABORT",
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                        ]
                    )
                logger.debug(
                    f"Sent abort notification for room {self.bootstrap_room} "
                    f"to {bootstrap_info.get('rank_ip', 'unknown')}:{bootstrap_info.get('rank_port', 'unknown')}"
                )
            except Exception as e:
                logger.debug(
                    f"Failed to send abort notification for room {self.bootstrap_room}: {e}"
                )


class CommonKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.pp_size = None
        self.attn_tp_size = None
        self.attn_cp_size = None
        self.dp_size = None
        self.page_size = None
        self.kv_cache_dtype: Optional[str] = None
        self.follow_bootstrap_room: Optional[bool] = None
        self.enable_dsa_cache_layer_split: Optional[bool] = None
        self.prefill_http_port: Optional[int] = None
        self.prefill_port_table: Dict[
            int, Dict[int, Dict[int, Dict[int, PrefillRankInfo]]]
        ] = {}
        self.room_to_dp_rank: Dict[int, Dict[str, Union[int, float]]] = {}
        self._registered_count = 0
        self.entry_cleanup_interval = (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL.get()
        )

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _is_ready(self) -> bool:
        if (
            self.attn_tp_size is None
            or self.attn_cp_size is None
            or self.pp_size is None
            or self.dp_size is None
        ):
            return False
        expected = self.dp_size * self.attn_cp_size * self.attn_tp_size * self.pp_size
        logger.debug(
            f"Expected {expected} prefill servers to be registered, {self._registered_count} registered so far"
        )
        return self._registered_count >= expected

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_post("/register_dp_rank", self._handle_register_dp_rank)
        self.app.router.add_post("/query_dp_ranks", self._handle_query_dp_ranks)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        attn_tp_size = data["attn_tp_size"]
        attn_tp_rank = data["attn_tp_rank"]
        attn_cp_size = data["attn_cp_size"]
        attn_cp_rank = data["attn_cp_rank"]
        attn_dp_size = data["attn_dp_size"]
        attn_dp_rank = data["attn_dp_rank"]
        pp_size = data["pp_size"]
        pp_rank = data["pp_rank"]
        system_dp_size = data["system_dp_size"]
        system_dp_rank = data["system_dp_rank"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        page_size = int(data["page_size"])
        kv_cache_dtype = data["kv_cache_dtype"]
        prefill_http_port = data.get("prefill_http_port")

        if self.attn_tp_size is None:
            self.attn_tp_size = attn_tp_size

        if self.attn_cp_size is None:
            self.attn_cp_size = attn_cp_size

        if self.dp_size is None:
            self.dp_size = attn_dp_size if system_dp_size == 1 else system_dp_size

        if self.pp_size is None:
            self.pp_size = pp_size

        if self.page_size is None and page_size is not None:
            self.page_size = page_size

        if self.kv_cache_dtype is None and kv_cache_dtype is not None:
            self.kv_cache_dtype = kv_cache_dtype

        if self.prefill_http_port is None and prefill_http_port is not None:
            self.prefill_http_port = int(prefill_http_port)

        if self.follow_bootstrap_room is None:
            load_balance_method = data.get(
                "load_balance_method", "follow_bootstrap_room"
            )
            self.follow_bootstrap_room = load_balance_method == "follow_bootstrap_room"

        if self.enable_dsa_cache_layer_split is None:
            self.enable_dsa_cache_layer_split = bool(
                data.get("enable_dsa_cache_layer_split", False)
            )

        if system_dp_size == 1:
            dp_group = attn_dp_rank
        else:
            dp_group = system_dp_rank

        # Add lock to make sure thread-safe
        async with self.lock:
            dp_group_table = self.prefill_port_table.setdefault(dp_group, {})
            cp_group_table = dp_group_table.setdefault(attn_cp_rank, {})
            tp_group_table = cp_group_table.setdefault(attn_tp_rank, {})

            tp_group_table[pp_rank] = PrefillRankInfo(
                rank_ip=rank_ip,
                rank_port=rank_port,
            )

            self._registered_count += 1

        expected = self.dp_size * self.attn_cp_size * self.attn_tp_size * self.pp_size
        logger.debug(
            f"Register prefill bootstrap: DP{dp_group} CP{attn_cp_rank} TP{attn_tp_rank} PP{pp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            f" ({self._registered_count}/{expected} registered)"
        )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        prefill_dp_rank = request.query.get("prefill_dp_rank")
        prefill_cp_rank = request.query.get("prefill_cp_rank")
        target_tp_rank = request.query.get("target_tp_rank")
        target_pp_rank = request.query.get("target_pp_rank")
        if (
            not prefill_dp_rank
            or not prefill_cp_rank
            or not target_tp_rank
            or not target_pp_rank
        ):
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        if (
            int(prefill_dp_rank) == -1
            and int(prefill_cp_rank) == -1
            and int(target_tp_rank) == -1
            and int(target_pp_rank) == -1
        ):
            if not self._is_ready():
                return web.Response(
                    text=f"Prefill server not fully registered yet"
                    f" ({self._registered_count} workers registered).",
                    status=503,
                )
            info = PrefillServerInfo(
                attn_tp_size=self.attn_tp_size,
                attn_cp_size=self.attn_cp_size,
                dp_size=self.dp_size,
                pp_size=self.pp_size,
                page_size=self.page_size,
                kv_cache_dtype=self.kv_cache_dtype,
                follow_bootstrap_room=(
                    self.follow_bootstrap_room
                    if self.follow_bootstrap_room is not None
                    else True
                ),
                enable_dsa_cache_layer_split=bool(self.enable_dsa_cache_layer_split),
                prefill_http_port=self.prefill_http_port,
            )
            return web.json_response(dataclasses.asdict(info), status=200)

        if not self._is_ready():
            return web.Response(
                text=f"Prefill server not fully registered yet"
                f" ({self._registered_count} workers registered).",
                status=503,
            )

        # Find corresponding prefill info
        try:
            async with self.lock:
                bootstrap_info = self.prefill_port_table[int(prefill_dp_rank)][
                    int(prefill_cp_rank)
                ][int(target_tp_rank)][int(target_pp_rank)]
        except KeyError:
            return web.Response(
                text=f"Bootstrap info not found for dp_rank={prefill_dp_rank} cp_rank={prefill_cp_rank} "
                f"tp_rank={target_tp_rank} pp_rank={target_pp_rank}",
                status=404,
            )

        return web.json_response(dataclasses.asdict(bootstrap_info), status=200)

    async def _handle_register_dp_rank(self, request: web.Request):
        data = await request.json()
        bootstrap_room = int(data["bootstrap_room"])
        dp_rank = int(data["dp_rank"])
        async with self.lock:
            self.room_to_dp_rank[bootstrap_room] = {
                "dp_rank": dp_rank,
                "timestamp": time.time(),
            }
        logger.debug(f"Registered dp_rank={dp_rank} for {bootstrap_room=}")
        return web.Response(text="OK", status=200)

    async def _handle_query_dp_ranks(self, request: web.Request):
        data = await request.json()
        bootstrap_rooms = data["bootstrap_rooms"]
        result = {}
        async with self.lock:
            for room in bootstrap_rooms:
                room_int = int(room)
                if room_int in self.room_to_dp_rank:
                    result[str(room_int)] = self.room_to_dp_rank[room_int]["dp_rank"]
        return web.json_response(result, status=200)

    async def _cleanup_expired_entries(self):
        """Remove entries older than cleanup interval from room_to_dp_rank."""
        while True:
            await asyncio.sleep(self.entry_cleanup_interval)
            current_time = time.time()
            async with self.lock:
                expired_keys = [
                    key
                    for key, value in self.room_to_dp_rank.items()
                    if current_time - value["timestamp"] > self.entry_cleanup_interval
                ]
                for key in expired_keys:
                    del self.room_to_dp_rank[key]
            if expired_keys:
                logger.debug(
                    f"Cleaned up {len(expired_keys)} expired entries from room_to_dp_rank"
                )

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._loop.create_task(self._cleanup_expired_entries())

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, host=self.host, port=self.port)
            self._loop.run_until_complete(site.start())
            logger.info(
                f"CommonKVBootstrapServer started successfully on {self.host}:{self.port}"
            )
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
