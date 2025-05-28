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
import time
from collections import defaultdict
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
from sglang.srt.utils import (
    get_free_port,
    get_int_env_var,
    get_ip,
    get_local_ip_by_remote,
)

logger = logging.getLogger(__name__)


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


class KVTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int64]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


# decode
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


# decode
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


class MooncakeKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        self.kv_args = args
        self.engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )
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
        self.request_status: Dict[int, KVPoll] = {}
        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_queue = queue.Queue()
            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
            self.start_prefill_thread()
            self._register_to_bootstrap()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()

            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            self.executor = concurrent.futures.ThreadPoolExecutor(
                get_int_env_var(
                    "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                    min(max(1, cpu_count // 8), 8),
                )
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(list)
            self.connection_lock = threading.Lock()
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                int(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2)), 1
            )
            self.start_decode_thread()
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.prefill_tp_size_table: Dict[str, int] = {}
            self.prefill_dp_size_table: Dict[str, int] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

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
        status = self.engine.transfer_sync(
            mooncake_session_id, prefill_aux_addr, decode_aux_addr, aux_item_len
        )
        return status

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int
    ):
        if ":" in remote:
            remote = remote.split(":")[0]
        self._connect("tcp://" + remote + ":" + str(dst_port)).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
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
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
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
                    reqs_to_be_processed = (
                        self.transfer_infos[kv_chunk.room].values()
                        if kv_chunk.room in self.transfer_infos
                        else []
                    )
                    polls = []
                    dst_ranks_infos = []
                    for req in reqs_to_be_processed:
                        if not req.is_dummy:
                            # Early exit if the request has failed
                            with self.session_lock:
                                if req.mooncake_session_id in self.failed_sessions:
                                    self.record_failure(
                                        kv_chunk.room,
                                        f"Decode instance could be dead, {req.mooncake_session_id} failed due to multiple errors",
                                    )
                                    self.update_status(kv_chunk.room, KVPoll.Failed)
                                    self.sync_status_to_decode_endpoint(
                                        req.endpoint,
                                        req.dst_port,
                                        req.room,
                                        KVPoll.Failed,
                                    )
                                    break

                            chunked_dst_kv_indice = req.dst_kv_indices[
                                kv_chunk.index_slice
                            ]

                            # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                            # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                            if len(chunked_dst_kv_indice) < len(
                                kv_chunk.prefill_kv_indices
                            ):
                                kv_chunk.prefill_kv_indices = (
                                    kv_chunk.prefill_kv_indices[
                                        len(chunked_dst_kv_indice)
                                    ]
                                )
                                logger.warning(
                                    f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                                )

                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                self.decode_kv_args_table[
                                    req.mooncake_session_id
                                ].dst_kv_ptrs,
                                chunked_dst_kv_indice,
                            )
                            if ret != 0:
                                with self.session_lock:
                                    self.session_failures[req.mooncake_session_id] += 1
                                    # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                    if (
                                        self.session_failures[req.mooncake_session_id]
                                        >= 1
                                    ):
                                        self.failed_sessions.add(
                                            req.mooncake_session_id
                                        )
                                        logger.error(
                                            f"Session {req.mooncake_session_id} failed."
                                        )
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint, req.dst_port, req.room, KVPoll.Failed
                                )
                                break

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
                                    status = (
                                        KVPoll.Success if all(polls) else KVPoll.Failed
                                    )
                                    self.update_status(req.room, status)
                                    for endpoint, dst_port, room in dst_ranks_infos:
                                        self.sync_status_to_decode_endpoint(
                                            endpoint, dst_port, room, status
                                        )
                        else:
                            # Dummy request means the decode instance is not used, so its status can be marked as success directly
                            # Dummy request does not need to sync status to decode endpoint
                            if kv_chunk.is_last and req.room in self.request_status:
                                self.update_status(req.room, KVPoll.Success)

                    if (
                        kv_chunk.room not in self.request_status
                        or self.check_status(kv_chunk.room) == KVPoll.Success
                    ):
                        self.transfer_infos.pop(kv_chunk.room)

                except queue.Empty:
                    continue
                except Exception as e:
                    # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                    raise RuntimeError(
                        f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                    )

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
                if status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get kvcache from prefill instance, it might be dead",
                    )
                self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

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

                            for bootstrap_room in self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ]:
                                # Remove KVPoll.Success requests from the map
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].remove(
                                        bootstrap_room
                                    )
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

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

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

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

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
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

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
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.kv_args.engine_rank,
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

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_tp_size_table:
                del self.prefill_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )


class MooncakeKVSender(BaseKVSender):

    def __init__(
        self, mgr: MooncakeKVManager, bootstrap_addr: str, bootstrap_room: int
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state = None
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
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()

        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)


class MooncakeKVReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

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
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None

        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            self.prefill_tp_size, self.prefill_dp_size = (
                self._get_prefill_parallel_info_from_server()
            )
            if self.prefill_tp_size is None or self.prefill_dp_size is None:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_tp_size}"
                )
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

        self.target_dp_group = self.bootstrap_room % self.prefill_dp_size

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
                    logger.debug(
                        f"Fetched bootstrap info: {bootstrap_info} for DP {self.target_dp_group} TP {target_tp_rank}"
                    )
                    bootstrap_infos.append(bootstrap_info)
                else:
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group}",
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
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].append(
            self.bootstrap_room
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
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

    def _get_prefill_parallel_info_from_server(self) -> Tuple[int, int]:
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
                return None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None

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
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()

        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)


class MooncakeKVBootstrapServer(BaseKVBootstrapServer):
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
        if self.tp_size_per_dp_rank is None:
            self.tp_size_per_dp_rank = tp_size_per_dp_rank

        if role == "Prefill":
            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}

            self.prefill_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register prefill bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
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

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
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
