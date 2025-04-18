from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
import socket
import struct
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
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_free_port, get_ip, get_local_ip_by_remote

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


@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int64]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_kv_indices: npt.NDArray[np.int64]
    dst_aux_ptrs: list[int]
    dst_aux_index: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_kv_indices=np.frombuffer(msg[5], dtype=np.int64),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            dst_aux_index=int(msg[7].decode("ascii")),
        )


class MooncakeKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ):
        self.engine = MooncakeTransferEngine()
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode
        # for p/d multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.request_status: Dict[int, KVPoll] = {}
        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_queue = queue.Queue()
            self.transfer_infos: Dict[int, TransferInfo] = {}
            self.start_prefill_thread()
            self._register_to_bootstrap()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.start_decode_thread()
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
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
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
    ):
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_args.kv_data_ptrs)
        for layer_id in range(num_layers):
            src_ptr = self.kv_args.kv_data_ptrs[layer_id]
            dst_ptr = dst_kv_ptrs[layer_id]
            item_len = self.kv_args.kv_item_lens[layer_id]

            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)

                # TODO: make async later
                status = self.engine.transfer_sync(
                    mooncake_session_id, src_addr, dst_addr, length
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
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                if room == "None":
                    continue
                room = int(room)
                self.transfer_infos[room] = TransferInfo.from_zmq(waiting_req_bytes)

                # NOTE: after bootstrapping we can mark the req as waiting for input
                self.request_status[room] = KVPoll.WaitingForInput

        def transfer_thread():
            # TODO: Shall we use KVPoll.Transferring state?
            while True:
                try:
                    kv_chunk: TransferKVChunk = self.transfer_queue.get(timeout=0.01)
                    req = self.transfer_infos[kv_chunk.room]
                    chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]
                    assert len(chunked_dst_kv_indice) == len(
                        kv_chunk.prefill_kv_indices
                    )

                    ret = self.send_kvcache(
                        req.mooncake_session_id,
                        kv_chunk.prefill_kv_indices,
                        req.dst_kv_ptrs,
                        chunked_dst_kv_indice,
                    )
                    if ret != 0:
                        self.request_status[kv_chunk.room] = KVPoll.Failed
                        self.sync_status_to_decode_endpoint(
                            req.endpoint, req.dst_port, req.room
                        )
                        continue

                    if kv_chunk.is_last:
                        # Only the last chunk we need to send the aux data
                        ret = self.send_aux(
                            req.mooncake_session_id,
                            kv_chunk.prefill_aux_index,
                            req.dst_aux_ptrs,
                            req.dst_aux_index,
                        )
                        self.request_status[req.room] = (
                            KVPoll.Success if ret == 0 else KVPoll.Failed
                        )
                        self.sync_status_to_decode_endpoint(
                            req.endpoint, req.dst_port, req.room
                        )
                        self.transfer_infos.pop(req.room)

                except queue.Empty:
                    continue

        threading.Thread(target=bootstrap_thread).start()
        threading.Thread(target=transfer_thread).start()

    def start_decode_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def decode_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        threading.Thread(target=decode_thread).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        self.transfer_queue.put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )
        )
        self.request_status[bootstrap_room] = KVPoll.WaitingForInput

    def check_status(self, bootstrap_room: int):
        # TOOD: do we really need the poll()?

        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: The prefill engine could recv bootstrapping first
            self.request_status[bootstrap_room] = max(
                self.request_status[bootstrap_room], status
            )

    def get_session_id(self):
        return self.engine.get_session_id()

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        if self.dist_init_addr:
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0])
        else:
            ip_address = get_ip()

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Prefill",
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.kv_args.engine_rank,
        }

        try:
            response = requests.put(url, json=payload)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill Failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(f"Prefill Failed to register to bootstrap server: {e}")


class MooncakeKVSender(BaseKVSender):

    def __init__(
        self, mgr: MooncakeKVManager, bootstrap_addr: str, bootstrap_room: int
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.session_id = self.kv_mgr.get_session_id()

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
    ):
        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room, kv_indices, index_slice, False
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
            )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class MooncakeKVReceiver(BaseKVReceiver):

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)

        # NOTE: key distinguished by bootstrap_addr and engine_rank
        bootstrap_key = f"{self.bootstrap_addr}_{self.kv_mgr.kv_args.engine_rank}"

        if bootstrap_key not in self.kv_mgr.connection_pool:
            self.bootstrap_info = self._get_bootstrap_info_from_server(
                self.kv_mgr.kv_args.engine_rank
            )
            if self.bootstrap_info is None:
                logger.error(
                    f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank}"
                )
            else:
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_info
        else:
            self.bootstrap_info = self.kv_mgr.connection_pool[bootstrap_key]

        assert self.bootstrap_info is not None
        self.kv_mgr.update_status(bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(self, engine_rank):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}"
            response = requests.get(url)
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

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None):
        self.prefill_server_url = (
            f"{self.bootstrap_info['rank_ip']}:{self.bootstrap_info['rank_port']}"
        )
        logger.debug(
            f"Fetched bootstrap info: {self.bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
        )

        packed_kv_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
        )
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )
        self._connect("tcp://" + self.prefill_server_url).send_multipart(
            [
                str(self.bootstrap_room).encode("ascii"),
                get_local_ip_by_remote().encode("ascii"),
                str(self.kv_mgr.rank_port).encode("ascii"),
                self.session_id.encode("ascii"),
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


class MooncakeKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.prefill_port_table: Dict[int, Dict[str, Union[str, int]]] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/metadata", self._handle_metadata)
        self.app.router.add_route("*", "/route", self._handle_route)

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
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        engine_rank = int(data["engine_rank"])

        # Add lock to make sure thread-safe
        if role == "Prefill":
            self.prefill_port_table[engine_rank] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Registered Prefill boostrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        if not engine_rank:
            return web.Response(text="Missing rank", status=400)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.prefill_port_table.get(int(engine_rank))

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
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
