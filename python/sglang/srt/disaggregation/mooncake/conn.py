from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import os
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

from sglang.srt.disaggregation.base.conn import BaseKVSender, KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    group_concurrent_contiguous,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_by_remote

logger = logging.getLogger(__name__)


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
    dst_kv_indices: npt.NDArray[np.int64]
    dst_aux_index: int
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int64)
            dst_aux_index = None
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int64)
            dst_aux_index = int(msg[5].decode("ascii"))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=int(msg[6].decode("ascii")),
            is_dummy=is_dummy,
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
        )


class MooncakeKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )
        self.request_status: Dict[int, KVPoll] = {}
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_queue = queue.Queue()
            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
            self.start_prefill_thread()

            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            self.executor = concurrent.futures.ThreadPoolExecutor(
                min(cpu_count // 4, 16)
            )
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

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_args.kv_data_ptrs)
        layers_params = [
            (
                self.kv_args.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                self.kv_args.kv_item_lens[layer_id],
            )
            for layer_id in range(num_layers)
        ]

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)

                status = self.engine.transfer_sync(
                    mooncake_session_id, src_addr, dst_addr, length
                )
                if status != 0:
                    return status
            return 0

        futures = [
            self.executor.submit(
                process_layer,
                src_ptr,
                dst_ptr,
                item_len,
            )
            for (src_ptr, dst_ptr, item_len) in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                # Immediate shutdown on first error (existing tasks will finish)
                self.executor.shutdown(wait=False)
                for f in futures:
                    f.cancel()
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
                str(self.check_status(room)).encode("ascii"),
            ]
        )

    def start_prefill_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[6].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        def transfer_thread():
            # TODO: Shall we use KVPoll.Transferring state?
            while True:
                try:
                    kv_chunk: TransferKVChunk = self.transfer_queue.get(timeout=0.01)
                    reqs_to_be_processed = self.transfer_infos[kv_chunk.room].values()
                    polls = []
                    dst_ranks_infos = []
                    for req in reqs_to_be_processed:
                        if not req.is_dummy:
                            chunked_dst_kv_indice = req.dst_kv_indices[
                                kv_chunk.index_slice
                            ]
                            assert len(chunked_dst_kv_indice) == len(
                                kv_chunk.prefill_kv_indices
                            ), f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"

                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                self.decode_kv_args_table[
                                    req.mooncake_session_id
                                ].dst_kv_ptrs,
                                chunked_dst_kv_indice,
                            )
                            if ret != 0:
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint, req.dst_port, req.room
                                )
                                continue

                            if kv_chunk.is_last:
                                # Only the last chunk we need to send the aux data
                                ret = self.send_aux(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_aux_index,
                                    self.decode_kv_args_table[
                                        req.mooncake_session_id
                                    ].dst_aux_ptrs,
                                    req.dst_aux_index,
                                )
                                polls.append(True if ret == 0 else False)
                                dst_ranks_infos.append(
                                    (req.endpoint, req.dst_port, req.room)
                                )

                                # Only sync status when all the dst ranks have received the kvcache
                                if len(polls) == req.required_dst_info_num:
                                    self.update_status(
                                        req.room,
                                        KVPoll.Success if all(polls) else KVPoll.Failed,
                                    )
                                    for endpoint, dst_port, room in dst_ranks_infos:
                                        self.sync_status_to_decode_endpoint(
                                            endpoint, dst_port, room
                                        )
                        else:
                            # Dummy request means the decode instance is not used, so its status can be marked as success directly
                            # Dummy request does not need to sync status to decode endpoint
                            if kv_chunk.is_last:
                                self.update_status(req.room, KVPoll.Success)

                    if self.check_status(kv_chunk.room) == KVPoll.Success:
                        self.transfer_infos.pop(kv_chunk.room)

                except queue.Empty:
                    continue

        threading.Thread(target=bootstrap_thread).start()
        threading.Thread(target=transfer_thread).start()

    def start_decode_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def decode_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.update_status(bootstrap_room, status)

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
        self.update_status(bootstrap_room, KVPoll.WaitingForInput)

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

    def get_session_id(self):
        return self.engine.get_session_id()


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
        # inner state
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

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
        # TODO: raise a real exception
        raise Exception("Fake KVSender Exception")


class MooncakeKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.kv_mgr = mgr
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.kv_mgr.update_status(bootstrap_room, KVPoll.WaitingForInput)

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                    ]
                )

    def init(self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            is_dummy = bootstrap_info["is_dummy"]

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        # TODO: raise a real exception
        raise Exception("Fake KVReceiver Exception")

class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    pass
