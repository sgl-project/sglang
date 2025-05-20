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


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


GUARD = "NixlMsgGuard".encode("ascii")


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    agent_metadata: bytes
    agent_name: str
    dst_kv_ptrs: list[int]
    dst_kv_indices: npt.NDArray[np.int64]
    dst_aux_ptrs: list[int]
    dst_aux_index: int
    dst_gpu_id: int
    required_dst_info_num: int

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
                agent_name="",
                dst_kv_ptrs=[],
                dst_kv_indices=np.array([], dtype=np.int64),
                dst_aux_ptrs=[],
                dst_aux_index=0,
                dst_gpu_id=0,
                required_dst_info_num=0,
            )
        else:
            return cls(
                room=int(msg[0].decode("ascii")),
                endpoint=msg[1].decode("ascii"),
                dst_port=int(msg[2].decode("ascii")),
                agent_metadata=msg[3],
                agent_name=msg[4].decode("ascii"),
                dst_kv_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
                dst_kv_indices=np.frombuffer(msg[6], dtype=np.int64),
                dst_aux_ptrs=list(struct.unpack(f"{len(msg[7])//8}Q", msg[7])),
                dst_aux_index=int(msg[8].decode("ascii")),
                dst_gpu_id=int(msg[9].decode("ascii")),
                required_dst_info_num=int(msg[10].decode("ascii")),
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
        self.is_mla_backend = is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        # for p/d multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        if not server_args.enable_dp_attention and server_args.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )
        # if self.enable_dp_attention:
        #     assert (
        #         server_args.dp_size > 1
        #     ), "If dp_attention is enabled, dp size must be greater than 1 in disaggregation mode."
        #     self.dp_size = server_args.dp_size
        #     self.tp_size_of_dp = server_args.tp_size // server_args.dp_size
        #     self.attn_tp_rank = args.engine_rank % self.tp_size_of_dp
        #     self.dp_rank = args.engine_rank // self.tp_size_of_dp

        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()

        self.rank_port = get_free_port()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.request_status = {}
            self.transfer_infos: Dict[int, TransferInfo] = {}
            self.peer_names: Dict[str, str] = {}
            self._start_bootstrap_thread()
            self._register_to_bootstrap()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # bootstrap key -> (remote_engine_rank -> possible remote source info)
            #self.prefill_peer_infos: Dict[str, list[Dict[int, NixlEngineInfo]]] = {}
            
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            self.prefill_tp_size_table: Dict[str, int] = {}
            self.prefill_dp_size_table: Dict[str, int] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: The prefill engine could recv bootstrapping first
            self.request_status[bootstrap_room] = max(
                self.request_status[bootstrap_room], status
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

    def _add_remote(self, agent_name: str, agent_metadata: bytes):
        if agent_name not in self.peer_names:
            self.peer_names[agent_name] = self.agent.add_remote_agent(agent_metadata)
        return self.peer_names[agent_name]

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

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        handles = []
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                return []

            peer_name = self._add_remote(req.agent_name, req.agent_metadata)
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
            handles.append(kv_xfer_handle)
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
                    is_last = bool(int(components[3]))
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
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
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
                required_dst_info_num = int(waiting_req_bytes[10].decode("ascii"))
                room = int(room)
                agent_name = waiting_req_bytes[4].decode("ascii")
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(waiting_req_bytes)
                
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

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
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

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
            return self.kv_mgr.check_status(self.bootstrap_room)
        states = [self.kv_mgr.agent.check_xfer_state(x) for x in self.xfer_handles]
        if all([x == "DONE" for x in states]):
            return KVPoll.Success  # type: ignore
        if any([x == "ERR" for x in states]):
            raise Exception("KVSender transfer encountered an error.")
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class NixlKVReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

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

        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            self.prefill_tp_size, self.prefill_dp_size = (
                self._get_prefill_dp_size_from_server()
            )
            if self.prefill_tp_size is None or self.prefill_dp_size is None:
                logger.error(
                    f"Could not fetch prefill parallel info for bootstrap_addr: {self.bootstrap_addr}"
                )
            else:
                self.kv_mgr.prefill_tp_size_table[self.bootstrap_addr] = (
                    self.prefill_tp_size
                )
                self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = (
                    self.prefill_dp_size
                )
        else:
            self.prefill_tp_size = self.kv_mgr.prefill_tp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_dp_size = self.kv_mgr.prefill_dp_size_table[
                self.bootstrap_addr
            ]
        # Currently, we don't allow prefill instance and decode instance to
        # have different TP sizes per DP rank, except for models using MLA.
        local_tp_size_per_dp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        prefill_tp_size_per_dp_rank = self.prefill_tp_size // self.prefill_dp_size
        if local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            ) // (local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank)
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            )
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks for non MLA models;
            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank + 1)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                )
            ]

            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1

        self.target_dp_group = bootstrap_room % self.prefill_dp_size

        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
                    # NOTE: only support MLA for now: select one prefill rank as real rank
                    bootstrap_info["is_dummy"] = not bool(
                        target_tp_rank == self.target_tp_rank
                        or self.target_tp_rank is None
                    )
                    bootstrap_infos.append(bootstrap_info)
                else:
                    logger.error(
                        f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group}"
                    )
            self.bootstrap_infos = bootstrap_infos

            if len(self.bootstrap_infos) == 0:
                logger.error(
                    f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank}"
                )
            else:
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos
                # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
                #self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
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

    def _get_prefill_dp_size_from_server(self) -> int:
        """Fetch the prefill parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return int(prefill_parallel_info["prefill_tp_size"]), int(
                    prefill_parallel_info["prefill_dp_size"]
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None

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
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            is_dummy = bootstrap_info["is_dummy"]

            # TODO: just send "" for indices for dummy
            if is_dummy:
                # TODO: need to set success??
                sock, lock = self._connect("tcp://" + self.prefill_server_url)
                with lock:
                    sock.send_multipart(
                        [
                            GUARD,
                            str(self.bootstrap_room).encode("ascii"),
                        ]
                    )
                continue
            
            # TODO: send_kv_args earlier
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            logger.debug(
                f"Sending to {self.prefill_server_url} with bootstrap room {self.bootstrap_room}"
            )
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        self.kv_mgr.agent.name.encode("ascii"),
                        packed_kv_data_ptrs,
                        kv_indices.tobytes(),
                        packed_aux_data_ptrs,
                        str(aux_index).encode("ascii"),
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
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
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.tp_size = None
        self.dp_size = None
        self.tp_size_per_dp_rank = None
        self.prefill_port_table: Dict[int, Dict[int, Dict[str, Union[str, int]]]] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)

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
        tp_size = data["tp_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        engine_rank = int(data["engine_rank"])

        if self.tp_size is None:
            self.tp_size = tp_size

        if self.dp_size is None:
            self.dp_size = dp_size

        tp_size_per_dp_rank = tp_size // dp_size
        if self.tp_size_per_dp_rank == None:
            self.tp_size_per_dp_rank = tp_size_per_dp_rank

        # Add lock to make sure thread-safe
        if role == "Prefill":
            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}

            self.prefill_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register Prefill bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            prefill_parallel_info = {
                "prefill_tp_size": self.tp_size,
                "prefill_dp_size": self.dp_size,
            }
            return web.json_response(prefill_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_group)][
                int(engine_rank)
            ]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

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
