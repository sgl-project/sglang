from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
import threading
import time
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    format_tcp_address,
    get_local_ip_auto,
    get_zmq_socket_on_host,
    is_valid_ipv6_address,
    maybe_wrap_ipv6_address,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PrefillServerInfo:
    attn_tp_size: int
    attn_cp_size: int
    dp_size: int
    pp_size: int
    page_size: Optional[int]
    kv_cache_dtype: Optional[str]
    follow_bootstrap_room: bool

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
        self.is_mla_backend = is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        self.server_args = server_args
        # for p/d multi node infer
        self.bootstrap_host = server_args.host
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_cp_size = get_attention_cp_size()
        self.attn_cp_rank = get_attention_cp_rank()
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

        # bind zmq socket
        context = zmq.Context()
        zmq_bind_host = maybe_wrap_ipv6_address(self.local_ip)
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=zmq_bind_host
        )
        logger.debug(f"kv manager bind to {zmq_bind_host}:{self.rank_port}")

        self.request_status: Dict[int, KVPoll] = {}
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # TODO(shangming): Fix me when we support MHA/GQA + CP, or when we utilize all cp ranks for KV transfer in CP mode.
            self.is_dummy_cp_rank = (
                is_mla_backend and self.attn_cp_size > 1 and self.attn_cp_rank != 0
            )
            self.register_to_bootstrap()
            self.transfer_infos = {}
            self.decode_kv_args_table = {}
            self.pp_group = get_pp_group()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            self.bootstrap_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
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
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def check_status(self, bootstrap_room: int) -> KVPoll:
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
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

    def ensure_parallel_info(
        self, bootstrap_addr: str, max_retries: int = 20, retry_interval: float = 1.0
    ) -> bool:
        """Fetch and cache prefill parallel info if not yet available.
        Returns True if info is available (cached or freshly fetched).
        Retries with backoff if the prefill server hasn't registered yet.
        """
        if bootstrap_addr in self.prefill_info_table:
            return True
        info = None
        for attempt in range(max_retries):
            info = self._fetch_prefill_server_info(bootstrap_addr)
            if info is not None:
                break
            if attempt < max_retries - 1:
                logger.info(
                    f"Prefill server info not available from {bootstrap_addr}, "
                    f"retrying ({attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_interval)
        if info is None:
            return False

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

        self.prefill_info_table[bootstrap_addr] = info
        logger.debug(f"Prefill parallel info for [{bootstrap_addr}]: {info}")
        return True

    @staticmethod
    def _fetch_prefill_server_info(
        bootstrap_addr: str,
    ) -> Optional[PrefillServerInfo]:
        """Fetch the prefill server info from the bootstrap server."""
        try:
            url = f"http://{bootstrap_addr}/route?prefill_dp_rank={-1}&prefill_cp_rank={-1}&target_tp_rank={-1}&target_pp_rank={-1}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return PrefillServerInfo(**data)
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill server info from bootstrap: {e}")
            return None

    def register_to_bootstrap(self):
        """Register prefill server info to bootstrap server via HTTP POST."""
        if self.dist_init_addr:
            # Multi-node case: bootstrap server's host is dist_init_addr
            if self.dist_init_addr.startswith("["):  # [ipv6]:port or [ipv6]
                if self.dist_init_addr.endswith("]"):
                    host = self.dist_init_addr
                else:
                    host, _ = self.dist_init_addr.rsplit(":", 1)
            else:
                host = socket.gethostbyname(self.dist_init_addr.rsplit(":", 1)[0])
        else:
            # Single-node case: bootstrap server's host is the same as http server's host
            host = self.bootstrap_host
            host = maybe_wrap_ipv6_address(host)

        bootstrap_server_url = f"{host}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
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
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Prefill instance failed to register to bootstrap server: {e}"
            )

    @cache
    def _connect(self, endpoint: str, is_ipv6: bool = False):
        socket = zmq.Context().socket(zmq.PUSH)
        if is_ipv6:
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(endpoint)
        return socket

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
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], int]:
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + len(src_kv_ptrs)
        if len(src_kv_ptrs) == len(dst_kv_ptrs):
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            # Decode pp size should be equal to prefill pp size or 1
            sliced_dst_kv_ptrs = dst_kv_ptrs[start_layer:end_layer]
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage


class CommonKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        # inner state
        self.curr_idx = 0
        if self.kv_mgr.is_dummy_cp_rank:
            # Non-authoritative CP ranks are dummy participants.
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        if (
            self.kv_mgr.server_args.dp_size > 1
            and self.kv_mgr.server_args.load_balance_method != "follow_bootstrap_room"
        ):
            self._register_prefill_dp_rank()

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

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        pass

    def poll(self) -> KVPoll:
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


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
        prefill_dp_rank: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

        if not self.kv_mgr.ensure_parallel_info(self.bootstrap_addr):
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            self.bootstrap_infos = None
            return

        self.prefill_info = self.kv_mgr.prefill_info_table[self.bootstrap_addr]

        # Rank mapping for PD with different TP sizes per rank for target DP/CP group
        if self.kv_mgr.attn_tp_size == self.prefill_info.attn_tp_size:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size
            )
            self.required_dst_info_num = 1
            self.required_prefill_response_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        elif self.kv_mgr.attn_tp_size > self.prefill_info.attn_tp_size:
            if not self.kv_mgr.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size
            ) // (self.kv_mgr.attn_tp_size // self.prefill_info.attn_tp_size)
            self.required_dst_info_num = (
                self.kv_mgr.attn_tp_size // self.prefill_info.attn_tp_size
            )
            self.required_prefill_response_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            if not self.kv_mgr.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks for non MLA models;
            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size)
                    * (self.prefill_info.attn_tp_size // self.kv_mgr.attn_tp_size),
                    (self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size + 1)
                    * (self.prefill_info.attn_tp_size // self.kv_mgr.attn_tp_size),
                )
            ]

            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1
            if self.kv_mgr.is_mla_backend:
                self.required_prefill_response_num = 1
            else:
                self.required_prefill_response_num = (
                    self.prefill_info.attn_tp_size // self.kv_mgr.attn_tp_size
                )

        # Decode cp size should be equal to prefill cp size or 1
        assert (
            self.kv_mgr.attn_cp_size == self.prefill_info.attn_cp_size
            or self.kv_mgr.attn_cp_size == 1
        ), (
            f"Decode cp size ({self.kv_mgr.attn_cp_size}) should be equal to prefill cp size ({self.prefill_info.attn_cp_size}) or 1",
        )
        if self.kv_mgr.attn_cp_size == self.prefill_info.attn_cp_size:
            self.target_cp_ranks = [self.kv_mgr.attn_cp_rank]
        else:
            self.target_cp_ranks = [
                rank for rank in range(self.prefill_info.attn_cp_size)
            ]
            # TODO(shangming): Support KVCache transfer for multiple prefill cp ranks -> 1 decode cp rank
            # For now, we handle the control plane in advance, but we need to support the data plane in the future.
            if self.kv_mgr.is_mla_backend:
                # For MLA: we only need to retrieve KVCache from the first CP rank now
                self.target_cp_ranks = self.target_cp_ranks[:1]
                self.required_prefill_response_num *= 1
            else:
                self.required_prefill_response_num *= (
                    self.prefill_info.attn_cp_size // self.kv_mgr.attn_cp_size
                )

        # Decode pp size should be equal to prefill pp size or 1
        assert (
            self.kv_mgr.pp_size == self.prefill_info.pp_size or self.kv_mgr.pp_size == 1
        ), (
            f"Decode pp size ({self.kv_mgr.pp_size}) should be equal to prefill pp size ({self.prefill_info.pp_size}) or 1",
        )
        if self.prefill_info.pp_size == self.kv_mgr.pp_size:
            self.target_pp_ranks = [self.kv_mgr.pp_rank]
        else:
            self.target_pp_ranks = [rank for rank in range(self.prefill_info.pp_size)]
            self.required_prefill_response_num *= (
                self.prefill_info.pp_size // self.kv_mgr.pp_size
            )

        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )

        assert (
            prefill_dp_rank is not None
        ), "prefill_dp_rank must be resolved before creating receiver"
        self.prefill_dp_rank = prefill_dp_rank
        self._setup_bootstrap_infos()

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
                            self.kv_mgr.update_status(
                                self.bootstrap_room, KVPoll.Failed
                            )
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
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @classmethod
    def _connect_to_bootstrap_server(cls, bootstrap_info: dict):
        ip_address = bootstrap_info["rank_ip"]
        port = bootstrap_info["rank_port"]
        is_ipv6_address = is_valid_ipv6_address(ip_address)
        sock, lock = cls._connect(
            format_tcp_address(ip_address, port), is_ipv6=is_ipv6_address
        )
        return sock, lock

    def _register_kv_args(self):
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


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

        if self.follow_bootstrap_room is None:
            load_balance_method = data.get(
                "load_balance_method", "follow_bootstrap_room"
            )
            self.follow_bootstrap_room = load_balance_method == "follow_bootstrap_room"

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
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
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
