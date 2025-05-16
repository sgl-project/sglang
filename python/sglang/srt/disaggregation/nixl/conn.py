from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
import socket
import struct
import threading
import uuid
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Set, Tuple, TypeAlias, Union

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
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_free_port, get_ip, get_local_ip_by_remote

logger = logging.getLogger(__name__)

NixlEngineInfo: TypeAlias = Dict[str, Union[str, int]]


# From Mooncake backend.
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


GUARD = "NixlMsgGuard".encode("ascii")


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_kv_indices: npt.NDArray[np.int64]
    dst_aux_ptrs: list[int]
    dst_aux_index: int
    dst_gpu_id: int

    def is_dummy(self):
        return self.endpoint == ""

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if len(msg) == 1:
            # dummy msg
            return cls(
                room=int(msg[0].decode("ascii")),
                endpoint="",
                dst_port=0,
                agent_metadata=b"",
                dst_kv_ptrs=[],
                dst_kv_indices=np.array([], dtype=np.int64),
                dst_aux_ptrs=[],
                dst_aux_index=0,
                dst_gpu_id=0,
            )
        else:
            return cls(
                room=int(msg[0].decode("ascii")),
                endpoint=msg[1].decode("ascii"),
                dst_port=int(msg[2].decode("ascii")),
                agent_metadata=msg[3],
                dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
                dst_kv_indices=np.frombuffer(msg[5], dtype=np.int64),
                dst_aux_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
                dst_aux_index=int(msg[7].decode("ascii")),
                dst_gpu_id=int(msg[8].decode("ascii")),
            )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunk IDs that have been received.
    received_kvs: Set[int] = dataclasses.field(default_factory=set)
    # Number of kv chunks to expect, will know this after last chunk is received.
    num_kvs_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False

    def is_done(self):
        if self.num_kvs_expected is None:
            return False
        return self.num_kvs_expected == len(self.received_kvs) and self.received_aux


class NixlKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        try:
            from nixl._api import nixl_agent
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e
        self.agent = nixl_agent(str(uuid.uuid4()))
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode
        # for p/d multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size

        self.tp_rank = args.engine_rank
        self.enable_dp_attention = server_args.enable_dp_attention
        if self.enable_dp_attention:
            assert (
                server_args.dp_size > 1
            ), "If dp_attention is enabled, dp size must be greater than 1 in disaggregation mode."
            self.dp_size = server_args.dp_size
            self.tp_size_of_dp = server_args.tp_size // server_args.dp_size
            self.attn_tp_rank = args.engine_rank % self.tp_size_of_dp
            self.dp_rank = args.engine_rank // self.tp_size_of_dp

        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()

        self.rank_port = get_free_port()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_infos: Dict[int, TransferInfo] = {}
            self.condition = threading.Condition()
            self.peer_names: Dict[int, str] = {}
            self._start_bootstrap_thread()
            self._register_to_bootstrap()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # bootstrap key -> (remote_engine_rank -> possible remote source info)
            self.prefill_peer_infos: Dict[str, list[Dict[int, NixlEngineInfo]]] = {}
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def register_buffer_to_engine(self):
        kv_addrs = []
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            kv_addrs.append((kv_data_ptr, kv_data_len, self.kv_args.gpu_id, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM", is_sorted=True)
        logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM", is_sorted=True)
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def _add_remote(self, room: int, agent_metadata: bytes):
        if room not in self.peer_names:
            self.peer_names[room] = self.agent.add_remote_agent(agent_metadata)
        return self.peer_names[room]

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        dst_gpu_id: int,
        notif: str,
    ):
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # Make descs
        num_layers = len(self.kv_args.kv_data_ptrs)
        src_addrs = []
        dst_addrs = []
        for layer_id in range(num_layers):
            src_ptr = self.kv_args.kv_data_ptrs[layer_id]
            dst_ptr = dst_kv_ptrs[layer_id]
            item_len = self.kv_args.kv_item_lens[layer_id]

            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                src_addrs.append((src_addr, length, self.kv_args.gpu_id))
                dst_addrs.append((dst_addr, length, dst_gpu_id))

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_kv_indices)}, after group: {len(src_addrs)}"
        )
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM", is_sorted=True)
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM", is_sorted=True)
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        # Make descs
        aux_item_len = self.kv_args.aux_item_lens[0]
        prefill_aux_addr = (
            self.kv_args.aux_data_ptrs[0] + prefill_aux_index * aux_item_len
        )
        decode_aux_addr = dst_aux_ptrs[0] + dst_aux_index * aux_item_len
        src_addrs = [(prefill_aux_addr, aux_item_len, 0)]
        dst_addrs = [(decode_aux_addr, aux_item_len, 0)]
        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM", is_sorted=True)
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM", is_sorted=True)
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        # Wait for transfer info to be populated by bootstrap thread.
        with self.condition:
            self.condition.wait_for(lambda: bootstrap_room in self.transfer_infos)
            req = self.transfer_infos[bootstrap_room]
        assert bootstrap_room == req.room

        if req.is_dummy():
            return []

        peer_name = self._add_remote(bootstrap_room, req.agent_metadata)
        chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
        assert len(chunked_dst_kv_indice) == len(kv_indices)

        notif = "_".join([str(req.room), "kv", str(chunk_id), str(int(is_last))])
        kv_xfer_handle = self.send_kvcache(
            peer_name,
            kv_indices,
            req.dst_kv_ptrs,
            chunked_dst_kv_indice,
            req.dst_gpu_id,
            notif,
        )
        handles = [kv_xfer_handle]
        # Only the last chunk we need to send the aux data.
        if is_last:
            assert aux_index is not None
            aux_xfer_handle = self.send_aux(
                peer_name,
                aux_index,
                req.dst_aux_ptrs,
                req.dst_aux_index,
                str(req.room) + "_aux",
            )
            handles.append(aux_xfer_handle)
        return handles

    def update_transfer_status(self):
        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            # We could also check that self.bootstrap_info['agent_name'] matches
            # the message sender. But the bootstrap room alone should be
            # sufficient to map the status.
            for msg in messages:
                components = msg.decode("ascii").split("_")
                room = int(components[0])
                if components[1] == "kv":
                    chunk_id = int(components[2])
                    is_last = bool(components[3])
                    self.transfer_statuses[room].received_kvs.add(chunk_id)
                    if is_last:
                        self.transfer_statuses[room].num_kvs_expected = chunk_id + 1
                elif components[1] == "aux":
                    self.transfer_statuses[room].received_aux = True

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

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
            "agent_name": self.agent.name,
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

    def _start_bootstrap_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                if room == "None":
                    continue
                room = int(room)
                with self.condition:
                    self.transfer_infos[room] = TransferInfo.from_zmq(waiting_req_bytes)
                    self.condition.notify_all()

        threading.Thread(target=bootstrap_thread).start()


class NixlKVSender(BaseKVSender):

    def __init__(self, mgr: NixlKVManager, bootstrap_addr: str, bootstrap_room: int):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.xfer_handles = []
        self.has_sent = False
        self.chunk_id = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
    ):
        new_xfer_handles = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
        )
        self.xfer_handles.extend(new_xfer_handles)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return KVPoll.WaitingForInput  # type: ignore
        states = [self.kv_mgr.agent.check_xfer_state(x) for x in self.xfer_handles]
        if all([x == "DONE" for x in states]):
            return KVPoll.Success  # type: ignore
        if any([x == "ERR" for x in states]):
            raise Exception("KVSender transfer encountered an error.")
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class NixlKVReceiver(BaseKVReceiver):

    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.started_transfer = False

        # NOTE: key distinguished by bootstrap_addr and engine_rank
        bootstrap_key = f"{self.bootstrap_addr}_{self.kv_mgr.kv_args.engine_rank}"

        if bootstrap_key not in self.kv_mgr.prefill_peer_infos:
            self.bootstrap_info = self._get_bootstrap_info_from_server(
                self.kv_mgr.kv_args.engine_rank
            )
            if self.bootstrap_info is None:
                logger.error(
                    f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank}"
                )
            else:
                self.kv_mgr.prefill_peer_infos[bootstrap_key] = self.bootstrap_info
        else:
            self.bootstrap_info = self.kv_mgr.prefill_peer_infos[bootstrap_key]
        assert self.bootstrap_info is not None

    # return a list of remotes in a dict, [(remote_engine_rank -> NixlEngineInfo), ...]
    # In each dict, there are multiple possible remotes named "equal sources".
    # We only need to select one to split the traffic. i.e. we totally select len(list) remotes.
    def _get_bootstrap_info_from_server(
        self, engine_rank
    ) -> Optional[List[Dict[int, NixlEngineInfo]]]:
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            if self.kv_mgr.enable_dp_attention:
                url = f"http://{self.bootstrap_addr}/route"
                response = requests.get(url)
                if response.status_code != 200:
                    logger.error(
                        f"Failed to get prefill server info: {response.status_code}, {response.text}"
                    )
                    return None

                bootstrap_info = response.json()
                assert isinstance(bootstrap_info, dict)
                bootstrap_info = {int(k): v for k, v in bootstrap_info.items()}

                # split out who need to send to this rank.
                # currently for dpsk mla model, those ranks share the same latent cache.
                # pick one as the real source

                prefill_tp_size = len(bootstrap_info.keys())

                assert (
                    prefill_tp_size >= self.kv_mgr.tp_size_of_dp
                ), f"Only support Prefill TP size >= Decode TP size of DP, now we have {prefill_tp_size} vs {self.kv_mgr.tp_size_of_dp}"

                num_remote_tp_rank_we_managed = (
                    prefill_tp_size // self.kv_mgr.tp_size_of_dp
                )

                # We handle [num * self.attn_tp_rank, num * self.attn_tp_rank + num)
                remote_tp_ranks = list(range(0, prefill_tp_size))
                # split it into tp_size_of_dp parts and get our part
                remote_tp_ranks_grouped = [
                    remote_tp_ranks[i : i + num_remote_tp_rank_we_managed]
                    for i in range(0, prefill_tp_size, self.kv_mgr.tp_size_of_dp)
                ]
                managed_ranks = remote_tp_ranks_grouped[self.kv_mgr.attn_tp_rank]

                assert len(managed_ranks) == num_remote_tp_rank_we_managed

                logger.debug(
                    f"Rank {self.kv_mgr.kv_args.engine_rank} source can be {managed_ranks}"
                )

                return [
                    {
                        rk: bootstrap_info[rk]
                        for rk in bootstrap_info.keys()
                        if rk in managed_ranks
                    }
                ]
            else:
                url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}"
                response = requests.get(url)
                if response.status_code == 200:
                    bootstrap_info = response.json()
                    return [{engine_rank: bootstrap_info}]
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

        assert self.bootstrap_info is not None
        assert self.bootstrap_room is not None

        for equal_sources in self.bootstrap_info:
            remote_rank = list(equal_sources.keys())[
                self.bootstrap_room % len(equal_sources)
            ]
            self.prefill_server_url = f"{equal_sources[remote_rank]['rank_ip']}:{equal_sources[remote_rank]['rank_port']}"
            logger.debug(
                f"Fetched bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank}, source: {remote_rank}, all: {list(equal_sources.keys())}"
            )

            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            logger.debug(
                f"Sending to {self.prefill_server_url} with bootstrap room {self.bootstrap_room}"
            )
            self._connect("tcp://" + self.prefill_server_url).send_multipart(
                [
                    GUARD,
                    str(self.bootstrap_room).encode("ascii"),
                    get_local_ip_by_remote().encode("ascii"),
                    str(self.kv_mgr.rank_port).encode("ascii"),
                    self.kv_mgr.agent.get_agent_metadata(),
                    packed_kv_data_ptrs,
                    kv_indices.tobytes(),
                    packed_aux_data_ptrs,
                    str(aux_index).encode("ascii"),
                    str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                ]
            )

            for dummy_rank in equal_sources.keys():
                if dummy_rank == remote_rank:
                    continue
                dummy_info = equal_sources[dummy_rank]
                dummy_url = f"{dummy_info['rank_ip']}:{dummy_info['rank_port']}"
                self._connect("tcp://" + dummy_url).send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                    ]
                )

        self.started_transfer = True

    def poll(self) -> KVPoll:
        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore

        self.kv_mgr.update_transfer_status()

        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            return KVPoll.Success  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class NixlKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, port: int):
        logger.debug(f"NixlKVBootstrapServer started on port {port}")
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
        agent_name = data["agent_name"]

        if role == "Prefill":
            async with self.lock:
                self.prefill_port_table[engine_rank] = {
                    "rank_ip": rank_ip,
                    "rank_port": rank_port,
                    "agent_name": agent_name,
                }
            logger.info(
                f"Registered Prefill boostrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port} and name: {agent_name}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        if not engine_rank:
            logger.debug(
                f"No engine_rank specified, return all {len(self.prefill_port_table)} engine infos as a dict"
            )
            # Return a dict of all engine_rank
            async with self.lock:
                bootstrap_info = self.prefill_port_table
            return web.json_response(bootstrap_info, status=200)

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
