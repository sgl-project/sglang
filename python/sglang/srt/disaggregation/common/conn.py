from __future__ import annotations

import asyncio
import logging
import socket
import threading
from functools import cache
from typing import Dict, List, Optional, Tuple, Union

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
from sglang.srt.layers.dp_attention import (
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
        # for p/d multi node infer
        self.bootstrap_host = server_args.host
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
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

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._register_to_bootstrap()
            self.transfer_infos = {}
            self.decode_kv_args_table = {}
            self.pp_group = get_pp_group()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.connection_lock = threading.Lock()
            self.required_prefill_response_num_table: Dict[int, int] = {}
            self.prefill_attn_tp_size_table: Dict[str, int] = {}
            self.prefill_dp_size_table: Dict[str, int] = {}
            self.prefill_pp_size_table: Dict[str, int] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
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
            "role": "Prefill",
            "attn_tp_size": self.attn_tp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_dp_size": self.attn_dp_size,
            "attn_dp_rank": self.attn_dp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "system_dp_size": self.system_dp_size,
            "system_dp_rank": self.system_dp_rank,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
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
        mgr: BaseKVManager,
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
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

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
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            (
                self.prefill_attn_tp_size,
                self.prefill_dp_size,
                self.prefill_pp_size,
            ) = self._get_prefill_parallel_info_from_server()
            if (
                self.prefill_attn_tp_size is None
                or self.prefill_dp_size is None
                or self.prefill_pp_size is None
            ):
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self.bootstrap_infos = None
                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_attn_tp_size} PP size:{self.prefill_pp_size}"
                )
                self.kv_mgr.prefill_attn_tp_size_table[self.bootstrap_addr] = (
                    self.prefill_attn_tp_size
                )
                self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = (
                    self.prefill_dp_size
                )
                self.kv_mgr.prefill_pp_size_table[self.bootstrap_addr] = (
                    self.prefill_pp_size
                )
        else:
            self.prefill_attn_tp_size = self.kv_mgr.prefill_attn_tp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_dp_size = self.kv_mgr.prefill_dp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_pp_size = self.kv_mgr.prefill_pp_size_table[
                self.bootstrap_addr
            ]

        # Handling for PD with different TP sizes per DP rank
        if self.kv_mgr.attn_tp_size == self.prefill_attn_tp_size:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size
            )
            self.required_dst_info_num = 1
            self.required_prefill_response_num = 1 * (
                self.prefill_pp_size // self.kv_mgr.pp_size
            )
            self.target_tp_ranks = [self.target_tp_rank]
        elif self.kv_mgr.attn_tp_size > self.prefill_attn_tp_size:
            if not self.kv_mgr.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size
            ) // (self.kv_mgr.attn_tp_size // self.prefill_attn_tp_size)
            self.required_dst_info_num = (
                self.kv_mgr.attn_tp_size // self.prefill_attn_tp_size
            )
            self.required_prefill_response_num = 1 * (
                self.prefill_pp_size // self.kv_mgr.pp_size
            )
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
                    * (self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size),
                    (self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size + 1)
                    * (self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size),
                )
            ]

            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1
            if self.kv_mgr.is_mla_backend:
                self.required_prefill_response_num = (
                    self.prefill_pp_size // self.kv_mgr.pp_size
                )
            else:
                self.required_prefill_response_num = (
                    self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size
                ) * (self.prefill_pp_size // self.kv_mgr.pp_size)

        if prefill_dp_rank is not None:
            logger.debug(f"Targeting DP rank: {prefill_dp_rank}")
            self.prefill_dp_rank = prefill_dp_rank
        else:
            self.prefill_dp_rank = bootstrap_room % self.prefill_dp_size

        self.target_dp_group = self.prefill_dp_rank

        # Decode pp size should be equal to prefill pp size or 1
        assert (
            self.kv_mgr.pp_size == self.prefill_pp_size or self.kv_mgr.pp_size == 1
        ), (
            f"Decode pp size ({self.kv_mgr.pp_size}) should be equal to prefill pp size ({self.prefill_pp_size}) or 1",
        )
        if self.prefill_pp_size == self.kv_mgr.pp_size:
            self.target_pp_ranks = [self.kv_mgr.pp_rank]
        else:
            self.target_pp_ranks = [rank for rank in range(self.prefill_pp_size)]

        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )
        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                # Enable higher PP ranks to be bootstrapped earlier to make PP PD requests bootstrap more robust
                for target_pp_rank in reversed(self.target_pp_ranks):
                    bootstrap_info = self._get_bootstrap_info_from_server(
                        target_tp_rank, self.target_dp_group, target_pp_rank
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
                            f"Fetched bootstrap info: {bootstrap_info} for DP {self.target_dp_group} TP {target_tp_rank} PP {target_pp_rank}"
                        )
                        bootstrap_infos.append(bootstrap_info)
                    else:
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group} and target_pp_rank {target_pp_rank}",
                        )
                        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                        return

            self.bootstrap_infos = bootstrap_infos
            self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

            # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
            self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0

    def _get_bootstrap_info_from_server(
        self, engine_rank, target_dp_group, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}&target_pp_rank={target_pp_rank}"
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

    def _get_prefill_parallel_info_from_server(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Fetch the prefill parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}&target_pp_rank={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return (
                    int(prefill_parallel_info["prefill_attn_tp_size"]),
                    int(prefill_parallel_info["prefill_dp_size"]),
                    int(prefill_parallel_info["prefill_pp_size"]),
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None, None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None, None

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
        self.dp_size = None
        self.prefill_port_table: Dict[
            int, Dict[int, Dict[int, Dict[str, Union[str, int]]]]
        ] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)
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
        role = data["role"]
        attn_tp_size = data["attn_tp_size"]
        attn_tp_rank = data["attn_tp_rank"]
        attn_dp_size = data["attn_dp_size"]
        attn_dp_rank = data["attn_dp_rank"]
        pp_size = data["pp_size"]
        pp_rank = data["pp_rank"]
        system_dp_size = data["system_dp_size"]
        system_dp_rank = data["system_dp_rank"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])

        if self.attn_tp_size is None:
            self.attn_tp_size = attn_tp_size

        if self.dp_size is None:
            self.dp_size = attn_dp_size if system_dp_size == 1 else system_dp_size

        if self.pp_size is None:
            self.pp_size = pp_size

        if role == "Prefill":
            if system_dp_size == 1:
                dp_group = attn_dp_rank
            else:
                dp_group = system_dp_rank

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}
                if attn_tp_rank not in self.prefill_port_table[dp_group]:
                    self.prefill_port_table[dp_group][attn_tp_rank] = {}

            self.prefill_port_table[dp_group][attn_tp_rank][pp_rank] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register prefill bootstrap: DP{dp_group} TP{attn_tp_rank} PP{pp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        target_pp_rank = request.query.get("target_pp_rank")
        if not engine_rank or not target_dp_group or not target_pp_rank:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if (
            int(engine_rank) == -1
            and int(target_dp_group) == -1
            and int(target_pp_rank) == -1
        ):
            prefill_parallel_info = {
                "prefill_attn_tp_size": self.attn_tp_size,
                "prefill_dp_size": self.dp_size,
                "prefill_pp_size": self.pp_size,
            }
            return web.json_response(prefill_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_group)][
                int(engine_rank)
            ][int(target_pp_rank)]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

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
