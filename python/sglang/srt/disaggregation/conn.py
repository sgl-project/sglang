from __future__ import annotations

import asyncio
import json
import logging
import random
import struct
import threading
from functools import cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.transfer_engine.mooncake import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode

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
    tp_size: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str


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
# TODO: To be obsoleted
# KVSENDER_POLLING_PORT = 17788
# KVRECEIVER_POLLING_PORT = 27788


class KVManager:
    # TODO: make it general and support multiple transfer backend before merging
    def __init__(self, args: KVArgs, disaggregation_mode: DisaggregationMode):
        self.engine = MooncakeTransferEngine()
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode
        self.request_pool: RequestPoolType = {}
        self.request_status: Dict[int, KVPoll] = {}
        self.connection_pool: Dict[int, str] = {}
        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
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

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

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

    def sync_status_to_decode_endpoint(self, remote: str, dst_port: int, room: int):
        if ":" in remote:
            remote = remote.split(":")[0]
        self._connect("tcp://" + remote + ":" + str(dst_port)).send_multipart(
            [
                str(room).encode("ascii"),
                str(self.request_status[room]).encode("ascii"),
            ]
        )

    def start_prefill_thread(self):
        self.rank_port = random.randint(20001, 25000)
        while True:
            try:
                self.server_socket.bind("tcp://*:" + str(self.rank_port))
                break
            except zmq.ZMQError as e:
                self.rank_port += 1

        def prefill_thread():
            while True:
                (
                    endpoint,
                    dst_port,
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
                dst_port = dst_port.decode("ascii")
                mooncake_session_id = mooncake_session_id.decode("ascii")
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                dst_ptrs = list(struct.unpack(f"{len(dst_ptrs)//8}Q", dst_ptrs))
                dst_kv_indices = np.frombuffer(dst_kv_indices, dtype=np.int64)
                dst_aux_ptrs = list(
                    struct.unpack(f"{len(dst_aux_ptrs)//8}Q", dst_aux_ptrs)
                )
                dst_aux_index = int(dst_aux_index.decode("ascii"))
                self.waiting_pool[bootstrap_room] = (
                    endpoint,
                    dst_port,
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
                        dst_port,
                        mooncake_session_id,
                        dst_ptrs,
                        dst_kv_indices,
                        dst_aux_ptrs,
                        dst_aux_index,
                    ) = self.waiting_pool.pop(room)
                    self.sync_status_to_decode_endpoint(endpoint, dst_port, room)
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
                        self.sync_status_to_decode_endpoint(endpoint, dst_port, room)
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
                    self.sync_status_to_decode_endpoint(endpoint, dst_port, room)

        threading.Thread(target=transfer_thread).start()

    def start_decode_thread(self):
        self.rank_port = random.randint(25001, 30000)
        while True:
            try:
                self.server_socket.bind("tcp://*:" + str(self.rank_port))
                break
            except zmq.ZMQError as e:
                self.rank_port += 1

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
        self.bootstrap_server_url = bootstrap_addr

        self.session_id = self.kv_mgr.get_session_id()

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        url = f"http://{self.bootstrap_server_url}/kv_route"
        payload = {
            "identity": self.session_id,
            "role": "Prefill",
            "serve_ip": self.bootstrap_server_url.split(":")[0],
            "serve_port": self.kv_mgr.rank_port,
            "tp_rank": self.kv_mgr.kv_args.engine_rank,
            "tp_size": self.kv_mgr.kv_args.tp_size,
        }

        logger.info(
            f"Register prefill server port {self.kv_mgr.rank_port} for tp_rank {self.kv_mgr.kv_args.engine_rank}"
        )

        try:
            response = requests.put(url, json=payload)
            if response.status_code == 200:
                logger.info(f"Prefill successfully registered to bootstrap server.")
            else:
                logger.info(
                    f"Prefill Failed to register to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.info(f"Prefill Failed to register to bootstrap server: {e}")

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index
        self.num_kv_indices = num_kv_indices

        # Register to bootstrap server
        logger.info(f"Prefill bootstrap addr {self.bootstrap_server_url}.")
        self._register_to_bootstrap()

    def send(self, kv_indices: npt.NDArray[np.int64]):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, self.aux_index)

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:

    def __init__(
        self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.decode_ip = self.kv_mgr.get_localhost()
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.set_status(bootstrap_room, KVPoll.WaitingForInput)
        self.prefill_engine_rank = None
        self.prefill_tp_size = None
        self.decode_port = self.kv_mgr.rank_port
        self.dealer_socket = None

    def _get_prefill_port_from_bootstrap(self, tp_rank: int):
        """Fetch the prefill server port corresponding to tp_rank from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/kv_route?tp_rank={tp_rank}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_serve_port = int(response.text)
                return prefill_serve_port
            else:
                logger.error(
                    f"Failed to get prefill server port: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill port from bootstrap: {e}")
            return None

    @cache
    def _connect_bootstrap(self, endpoint: str):
        socket = zmq.Context().socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, self.session_id)
        socket.connect(endpoint)
        return socket

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None):
        prefill_port = None
        logger.info(f"Decode bootstrap addr {self.bootstrap_addr}.")

        if self.kv_mgr.kv_args.engine_rank not in self.kv_mgr.connection_pool:
            prefill_port = self._get_prefill_port_from_bootstrap(
                self.kv_mgr.kv_args.engine_rank
            )
            if prefill_port is None:
                logger.error(
                    f"Could not fetch prefill server port for tp_rank {self.kv_mgr.kv_args.engine_rank}"
                )
            else:
                self.kv_mgr.connection_pool[self.kv_mgr.kv_args.engine_rank] = (
                    prefill_port
                )
        else:
            prefill_port = self.kv_mgr.connection_pool[self.kv_mgr.kv_args.engine_rank]

        self.prefill_server_url = f"{self.bootstrap_addr.split(':')[0]}:{prefill_port}"

        if prefill_port is not None:
            logger.info(
                f"Fetched prefill server port {prefill_port} for tp_rank {self.kv_mgr.kv_args.engine_rank}"
            )
            self.handshake_prefill_server(kv_indices, aux_index)

    def handshake_prefill_server(
        self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None
    ):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, aux_index)
        packed_kv_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
        )
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )
        self._connect("tcp://" + self.prefill_server_url).send_multipart(
            [
                self.decode_ip.encode("ascii"),
                str(self.decode_port).encode("ascii"),
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
        self.lock = asyncio.Lock()
        self._setup_routes()
        # prefill_engine_rank -> prefill_serve_port
        self.prefill_port_table: Dict[int, int] = {}

        self.context = zmq.Context()

        self.prefill_engine_rank = None
        self.prefill_tp_size = None

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/metadata", self._handle_metadata)
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
        method = request.method
        if method == "PUT":
            return await self._handle_kv_route_put(request)
        elif method == "GET":
            return await self._handle_kv_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_kv_route_put(self, request: web.Request):
        data = await request.json()
        # TODO: validate request
        # if not data or 'serve_port' not in data or 'tp_size' not in data:
        #     return web.Response(
        #         text="Missing information", status=400
        #     )
        identity = data["identity"]
        role = data["role"]
        serve_ip = data["serve_ip"]
        serve_port = int(data["serve_port"])  # Assuming serve_port is an integer
        tp_size = int(data["tp_size"])
        tp_rank = int(data["tp_rank"])

        # Add lock to make sure thread-safe
        if role == "Prefill":
            async with self.lock:
                self.prefill_port_table[tp_rank] = serve_port
            print(f"Registered Prefill tp_rank: {tp_rank} serve_port: {serve_port}")

        return web.Response(text="OK", status=200)

    async def _handle_kv_route_get(self, request: web.Request):
        tp_rank = request.query.get("tp_rank")
        if not tp_rank:
            return web.Response(text="Missing tp_rank", status=400)
        try:
            tp_rank = int(tp_rank)
        except ValueError:
            return web.Response(text="tp_rank must be int", status=400)

        # Find corresponding port
        async with self.lock:
            prefill_serve_port = self.prefill_port_table.get(tp_rank)

        if prefill_serve_port is not None:
            return web.Response(text=str(prefill_serve_port), status=200)
        else:
            return web.Response(text="Not Found", status=404)

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

    def poll(self) -> KVPoll: ...
