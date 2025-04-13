from __future__ import annotations

import asyncio
import logging
import struct
import threading
from functools import cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zmq, socket
from aiohttp import web
import requests

from sglang.srt.disaggregation.transfer_engine.mooncake import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_free_port

logger = logging.getLogger(__name__)


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    src_groups = []
    dst_groups = []
    current_src = [src_indices[0]]
    current_dst = [dst_indices[0]]

    for i in range(1, len(src_indices)):
        src_contiguous = src_indices[i] == src_indices[i - 1] + 1
        dst_contiguous = dst_indices[i] == dst_indices[i - 1] + 1
        if src_contiguous and dst_contiguous:
            current_src.append(src_indices[i])
            current_dst.append(dst_indices[i])
        else:
            src_groups.append(current_src)
            dst_groups.append(current_dst)
            current_src = [src_indices[i]]
            current_dst = [dst_indices[i]]

    src_groups.append(current_src)
    dst_groups.append(current_dst)

    return src_groups, dst_groups


class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str
    tp_size: int


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


RequestPoolType = Dict[int, Tuple[npt.NDArray[np.int64], Optional[int]]]
WaitingPoolType = Dict[
    int, Tuple[str, list[int], npt.NDArray[np.int64], list[int], int]
]
KVSENDER_POLLING_PORT = 17788
KVRECEIVER_POLLING_PORT = 27788


class KVManager:
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    # TODO: make it general and support multiple transfer backend before merging
    def __init__(self, args: KVArgs, disaggregation_mode: DisaggregationMode, server_args: ServerArgs):
        self.engine = MooncakeTransferEngine()
        self.kv_args = args
        self.server_args = server_args
        self.disaggregation_mode = disaggregation_mode
        self.request_pool: RequestPoolType = {}
        self.request_status: Dict[int, KVPoll] = {}
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        self.rank_port = None

        # TODO: in decode node, this is no need to be called   in the future
        # For now, prefill mulit-node need use the same prefill addr registerd to bootstrap server
        self.prefill_addr = self.parse_prefill_addr(server_args)

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.waiting_pool: WaitingPoolType = {}
            self.transfer_event = threading.Event()
            self.start_prefill_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.start_decode_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def register_buffer_to_engine(self):
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            self.engine.register(kv_data_ptr, kv_data_len)

        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            self.engine.register(aux_data_ptr, aux_data_len)

    def register_zmq_info_to_bootstrap_server(self, key, value, bootstrap_addr: str = None):
        """
        register to the bootstrap server
        """
        bootstrap_url = self.parse_bootstrap_addr(self.server_args, bootstrap_addr)
        respo = requests.put(bootstrap_url, params={"key": key}, json=value)
        if respo.status_code != 200:
            raise Exception("error registering zmq info to bootstrap server")

    def parse_prefill_addr(self, args: ServerArgs):
        port = args.port  # http_server_port
        dist_init_addr = args.dist_init_addr
        if dist_init_addr:
            ip_address = socket.gethostbyname(dist_init_addr.split(":")[0])
        else:
            ip_address = self.engine.get_localhost()
        return f"{ip_address}:{port}"

    def parse_bootstrap_addr(self, args: ServerArgs, bootstrap_addr: str = ""):
        """
        parse the bootstrap addr from the server args
        """
        scheme = "http://"

        if bootstrap_addr:
            # for now bootstrap_port is fixed to 8998
            return f"{scheme}{bootstrap_addr}/kv_route"

        else:
            port = args.disaggregation_bootstrap_port  # bootstrap_port
            dist_init_addr = args.dist_init_addr
            if dist_init_addr:
                ip_address = socket.gethostbyname(dist_init_addr.split(":")[0])
            else:
                ip_address = self.engine.get_localhost()
            return f"{scheme}{ip_address}:{port}/kv_route"

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @cache
    def query_zmq_rank_addr(self, key, bootstrap_url):
        resp = requests.get(bootstrap_url, params={"key": key})
        if resp.status_code != 200:
            raise Exception("Cant query receiver rank port for key {}, resp status {}".format(key, resp.status_code))
        ip, port = resp.json()['zmq_ip'], resp.json()['zmq_port']
        return ip, port

    @cache
    def get_free_zmq_port(self, key):
        return get_free_port()

    @cache
    def get_prefill_register_key(self, prefill_addr: str):
        key = f'{prefill_addr}_{self.kv_args.tp_size}_{self.kv_args.engine_rank}'
        return key

    @cache
    def get_decode_register_key(self, session_id):
        key = f'{session_id}_{self.kv_args.tp_size}_{self.kv_args.engine_rank}'
        return key

    @cache
    def get_pd_meta_key(self):
        """
        """
        return f"{self.engine.session_id}_{self.disaggregation_mode}_{self.kv_args.tp_size}_{self.kv_args.engine_rank}"

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
    ):
        layer_num = int(len(self.kv_args.kv_data_ptrs) / 2)
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )
        for layer_id in range(layer_num):
            prefill_key_layer_ptr = self.kv_args.kv_data_ptrs[layer_id]
            key_item_len = self.kv_args.kv_item_lens[layer_id]
            prefill_value_layer_ptr = self.kv_args.kv_data_ptrs[layer_num + layer_id]
            value_item_len = self.kv_args.kv_item_lens[layer_num + layer_id]

            decode_key_layer_ptr = dst_ptrs[layer_id]
            decode_value_layer_ptr = dst_ptrs[layer_num + layer_id]

            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                prefill_key_addr = (
                    prefill_key_layer_ptr + int(prefill_index[0]) * key_item_len
                )
                decode_key_addr = (
                    decode_key_layer_ptr + int(decode_index[0]) * key_item_len
                )

                # TODO: mooncake transfer engine can do async transfer. Do async later
                status = self.engine.transfer_sync(
                    mooncake_session_id,
                    prefill_key_addr,
                    decode_key_addr,
                    key_item_len * len(prefill_index),
                )
                if status != 0:
                    return status

                prefill_value_addr = (
                    prefill_value_layer_ptr + int(prefill_index[0]) * value_item_len
                )

                decode_value_addr = (
                    decode_value_layer_ptr + int(decode_index[0]) * value_item_len
                )

                # TODO: mooncake transfer engine can do async transfer. Do async later
                status = self.engine.transfer_sync(
                    mooncake_session_id,
                    prefill_value_addr,
                    decode_value_addr,
                    value_item_len * len(prefill_index),
                )
                if status != 0:
                    return status
        return 0

    def send_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
    ):
        aux_item_len = self.kv_args.aux_item_lens[0]
        prefill_aux_addr = (
            self.kv_args.aux_data_ptrs[0] + prefill_aux_index * aux_item_len
        )
        decode_aux_addr = dst_aux_ptrs[0] + dst_aux_index * aux_item_len
        # TODO: mooncake transfer engine can do async transfer. Do async later
        # Not sure about the amount of aux data, maybe transfer it by zmq is more effective
        status = self.engine.transfer_sync(
            mooncake_session_id, prefill_aux_addr, decode_aux_addr, aux_item_len
        )
        return status

    def sync_status_to_decode_endpoint(self, remote: str, room: int, mooncake_session_id: str):
        if ":" in remote:
            remote = remote.split(":")[0]
        query_key = self.get_decode_register_key(mooncake_session_id)
        bootstrap_addr = self.parse_bootstrap_addr(self.server_args)

        _, receiver_rank_port = self.query_zmq_rank_addr(query_key, bootstrap_addr)
        sock, lock = self._connect(
            "tcp://"
            + remote
            + ":"
            + str(receiver_rank_port)
        )
        with lock:
            sock.send_multipart(
                [
                    str(room).encode("ascii"),
                    str(self.request_status[room]).encode("ascii"),
                ]
            )

    def start_prefill_thread(self):
        self.rank_port = self.get_free_zmq_port(self.get_pd_meta_key())
        # should register to the bootstrap server, now to metadata server
        self.register_zmq_info_to_bootstrap_server(
            self.get_prefill_register_key(self.prefill_addr),
            {"zmq_port": self.rank_port, "zmq_ip": self.engine.get_localhost()}
        )
        self.server_socket.bind("tcp://*:" + str(self.rank_port))

        def prefill_thread():
            while True:
                (
                    endpoint,
                    mooncake_session_id,
                    bootstrap_room,
                    dst_ptrs,
                    dst_kv_indices,
                    dst_aux_ptrs,
                    dst_aux_index,
                ) = self.server_socket.recv_multipart()
                if bootstrap_room.decode("ascii") == "None":
                    continue
                endpoint = endpoint.decode("ascii")
                mooncake_session_id = mooncake_session_id.decode("ascii")
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                dst_ptrs = list(struct.unpack(f"{len(dst_ptrs) // 8}Q", dst_ptrs))
                dst_kv_indices = np.frombuffer(dst_kv_indices, dtype=np.int64)
                dst_aux_ptrs = list(
                    struct.unpack(f"{len(dst_aux_ptrs) // 8}Q", dst_aux_ptrs)
                )
                dst_aux_index = int(dst_aux_index.decode("ascii"))
                self.waiting_pool[bootstrap_room] = (
                    endpoint,
                    mooncake_session_id,
                    dst_ptrs,
                    dst_kv_indices,
                    dst_aux_ptrs,
                    dst_aux_index,
                )
                self.transfer_event.set()

        threading.Thread(target=prefill_thread).start()

        def transfer_thread():
            while True:
                self.transfer_event.wait()
                self.transfer_event.clear()
                bootstrap_room_ready = self.request_pool.keys()
                bootstrap_room_request = self.waiting_pool.keys()
                for room in list(bootstrap_room_request):
                    if room not in list(bootstrap_room_ready):
                        continue
                    status = KVPoll.Transferring
                    self.request_status[room] = status
                    (
                        endpoint,
                        mooncake_session_id,
                        dst_ptrs,
                        dst_kv_indices,
                        dst_aux_ptrs,
                        dst_aux_index,
                    ) = self.waiting_pool.pop(room)
                    self.sync_status_to_decode_endpoint(endpoint, room, mooncake_session_id)
                    (
                        prefill_kv_indices,
                        prefill_aux_index,
                    ) = self.request_pool.pop(room)
                    ret = self.send_kvcache(
                        mooncake_session_id,
                        prefill_kv_indices,
                        dst_ptrs,
                        dst_kv_indices,
                    )
                    if ret != 0:
                        status = KVPoll.Failed
                        self.sync_status_to_decode_endpoint(endpoint, room, mooncake_session_id)
                        continue
                    ret = self.send_aux(
                        mooncake_session_id,
                        prefill_aux_index,
                        dst_aux_ptrs,
                        dst_aux_index,
                    )
                    if ret != 0:
                        status = KVPoll.Failed
                    else:
                        status = KVPoll.Success
                    self.request_status[room] = status
                    self.sync_status_to_decode_endpoint(endpoint, room, mooncake_session_id)

        threading.Thread(target=transfer_thread).start()

    def start_decode_thread(self):
        self.rank_port = self.get_free_zmq_port(self.get_pd_meta_key())
        self.server_socket.bind("tcp://*:" + str(self.rank_port))

        def decode_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        threading.Thread(target=decode_thread).start()

    def enqueue_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        aux_index: Optional[int],
    ):
        self.request_pool[bootstrap_room] = (kv_indices, aux_index)
        self.request_status[bootstrap_room] = KVPoll.WaitingForInput
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_event.set()

    def check_status(self, bootstrap_room: int):
        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.request_status[bootstrap_room] == KVPoll.Success
        ):
            if bootstrap_room in self.request_pool:
                self.request_pool.pop(bootstrap_room)

        return self.request_status[bootstrap_room]

    def set_status(self, bootstrap_room: int, status: KVPoll):
        self.request_status[bootstrap_room] = status

    def get_localhost(self):
        return self.engine.get_localhost()

    def get_session_id(self):
        return self.engine.get_session_id()


class KVSender:

    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.set_status(bootstrap_room, KVPoll.WaitingForInput)
        self.aux_index = None

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index
        self.num_kv_indices = num_kv_indices

    def send(self, kv_indices: npt.NDArray[np.int64]):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, self.aux_index)

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self, mgr: KVManager, prefill_addr: str, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.bootstrap_url = self.kv_mgr.parse_bootstrap_addr(mgr.server_args, bootstrap_addr)
        prefill_ip, sender_rank_port = self.kv_mgr.query_zmq_rank_addr(
            self.kv_mgr.get_prefill_register_key(prefill_addr), self.bootstrap_url)

        logger.debug(f"KVReceiver init: prefill_addr={prefill_addr}, sender_rank_port={sender_rank_port}")
        self.prefill_server_url = (
            prefill_ip
            + ":"
            + str(sender_rank_port)
        )
        self.register_to_bootstrap(self.prefill_server_url)

        self.decode_ip = self.kv_mgr.get_localhost()
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.set_status(bootstrap_room, KVPoll.WaitingForInput)

    @cache
    def register_to_bootstrap(self, prefill_addr: str) -> None:
        logger.debug(f"KVReceiver Registering prefill_addr={prefill_addr}")
        # should register to the bootstrap server, now to metadata server
        self.kv_mgr.register_zmq_info_to_bootstrap_server(
            self.kv_mgr.get_decode_register_key(self.kv_mgr.engine.get_session_id()),
            {"zmq_port": self.kv_mgr.rank_port, "zmq_ip": self.kv_mgr.engine.get_localhost()}, bootstrap_addr=self.bootstrap_addr)

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    def init(self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, aux_index)
        packed_kv_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
        )
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )
        socket, lock = self._connect("tcp://" + self.prefill_server_url)
        with lock:
            socket.send_multipart(
                [
                    self.decode_ip.encode("ascii"),
                    self.session_id.encode("ascii"),
                    str(self.bootstrap_room).encode("ascii"),
                    packed_kv_data_ptrs,
                    kv_indices.tobytes(),
                    packed_aux_data_ptrs,
                    str(aux_index).encode("ascii"),
                ]
            )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.kv_route_store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        # will deprecate in the future
        self.app.router.add_route("*", "/metadata", self._handle_metadata)

        # only route for bootstrap server
        self.app.router.add_route("*", "/kv_route", self._handle_kv_route)

    async def _handle_metadata(self, request: web.Request):
        key = request.query.get("key", "")

        if request.method == "GET":
            return await self._handle_metadata_get(key)
        elif request.method == "PUT":
            return await self._handle_metadata_put(key, request)
        elif request.method == "DELETE":
            return await self._handle_metadata_delete(key)
        return web.Response(
            text="Method not allowed", status=405, content_type="application/json"
        )

    async def _handle_metadata_get(self, key):
        async with self.lock:
            value = self.store.get(key)
        if value is None:
            return web.Response(
                text="metadata not found", status=404, content_type="application/json"
            )
        return web.Response(body=value, status=200, content_type="application/json")

    async def _handle_metadata_put(self, key, request):
        data = await request.read()
        async with self.lock:
            self.store[key] = data
        return web.Response(
            text="metadata updated", status=200, content_type="application/json"
        )

    async def _handle_metadata_delete(self, key):
        async with self.lock:
            if key not in self.store:
                return web.Response(
                    text="metadata not found",
                    status=404,
                    content_type="application/json",
                )
            del self.store[key]
        return web.Response(
            text="metadata deleted", status=200, content_type="application/json"
        )

    async def _handle_kv_route(self, request: web.Request):
        key = request.query.get("key", "")

        if request.method == "GET":
            return await self._handle_kv_route_get(key)
        elif request.method == "PUT":
            return await self._handle_kv_route_put(key, request)
        elif request.method == "DELETE":
            return await self._handle_metadata_delete(key)
        return web.Response(
            text="Method not allowed", status=405, content_type="application/json"
        )

    async def _handle_kv_route_put(self, key, request: web.Request):
        data = await request.read()
        async with self.lock:
            self.kv_route_store[key] = data
        return web.Response(
            text="kv route info updated", status=200, content_type="application/json"
        )

    async def _handle_kv_route_get(self, key):
        async with self.lock:
            value = self.kv_route_store.get(key)
        if value is None:
            return web.Response(
                text="metadata not found", status=404, content_type="application/json"
            )
        return web.Response(body=value, status=200, content_type="application/json")

    async def _handle_kv_route_delete(self, key):
        async with self.lock:
            if key not in self.kv_route_store:
                return web.Response(
                    text="metadata not found",
                    status=404,
                    content_type="application/json",
                )
            del self.kv_route_store[key]
        return web.Response(
            text="metadata deleted", status=200, content_type="application/json"
        )

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._runner = web.AppRunner(self.app)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
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

    def poll(self) -> KVPoll:
        ...
