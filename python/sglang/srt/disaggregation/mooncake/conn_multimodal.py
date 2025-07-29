from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
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
import torch
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
from sglang.srt.disaggregation.common.utils import FastQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_free_port,
    get_int_env_var,
    get_ip,
    get_local_ip_by_remote,
)

logger = logging.getLogger(__name__)


class EmbeddingTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"EmbeddingTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


@dataclasses.dataclass
class TransferEmbeddingChunk:
    room: int
    embedding_index: int
    is_last: bool
    chunk_info: List[Tuple[int, int]]


@dataclasses.dataclass
class TransferEmbeddingInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_index: int
    required_dst_info_num: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_index=int(msg[4].decode("ascii")),
            required_dst_info_num=int(msg[5].decode("ascii")),
        )


@dataclasses.dataclass
class EmbeddingArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_ptrs: list[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
        )


class MooncakeEmbeddingManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ):
        self.data_args = args
        self.engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=self.data_args.gpu_id,
            ib_device=self.data_args.ib_device,
        )
        logger.debug(
            f"init MooncakeTransferEngine: {get_local_ip_by_remote()=};{self.data_args.gpu_id=};{self.data_args.ib_device=}"
        )
        self.disaggregation_mode = disaggregation_mode
        # for embedding/language model multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.request_status: Dict[int, KVPoll] = {}
        self.rank_port = None
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.EMBEDDING:
            self.transfer_infos: Dict[int, Dict[str, TransferEmbeddingInfo]] = {}
            self.language_args_table: Dict[str, EmbeddingArgsRegisterInfo] = {}
            self.start_embedding_thread()
            self._register_to_bootstrap()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()

            # Determine the number of threads to use for aux sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                min(max(4, int(0.75 * cpu_count) // 8), 12),
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4)
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            for queue, executor in zip(self.transfer_queues, self.executors):
                threading.Thread(
                    target=self.transfer_worker, args=(queue, executor), daemon=True
                ).start()

            self.bootstrap_time_out = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 30
            )
        elif self.disaggregation_mode == DisaggregationMode.LANGUAGE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            self.connection_lock = threading.Lock()
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_language_thread()
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.embedding_tp_size_table: Dict[str, int] = {}
            self.embedding_dp_size_table: Dict[str, int] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def register_buffer_to_engine(self):
        # Only register aux data buffers, not kv data buffers
        for aux_data_ptr, aux_data_len in zip(
            self.data_args.aux_data_ptrs, self.data_args.aux_data_lens
        ):
            self.engine.register(aux_data_ptr, aux_data_len)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def send_embedding(
        self,
        mooncake_session_id: str,
        embedding_index: int,
        dst_embedding_ptrs: list[int],
        dst_embedding_index: int,
        chunk_info: List[Tuple[int, int]],
    ):

        status_list = []

        for i in range(len(self.data_args.aux_item_lens)):
            chunk_offset, chunk_size = chunk_info[i]
            embedding_item_len = self.data_args.aux_item_lens[i]
            embedding_addr = (
                self.data_args.aux_data_ptrs[i] + embedding_index * embedding_item_len + chunk_offset
            )
            dst_embedding_addr = (
                dst_embedding_ptrs[i] + dst_embedding_index * embedding_item_len + chunk_offset
            )

            status = self.engine.transfer_sync(
                mooncake_session_id,
                embedding_addr,
                dst_embedding_addr,
                chunk_size,
            )
            status_list.append(status)

        return 0 if sum(status_list) == 0 else 1

    def sync_status_to_language_endpoint(
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

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                embedding_chunk: TransferEmbeddingChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[embedding_chunk.room].values()
                    if embedding_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                logger.debug(
                    f"Transfer worker received request with {embedding_chunk.room=};{embedding_chunk.embedding_index=};{embedding_chunk.is_last=}"
                )

                for req in reqs_to_be_processed:
                    logger.debug(
                        f"Transfer worker received request with {req.mooncake_session_id=}"
                    )

                    # Early exit if the request has failed
                    with self.session_lock:
                        if req.mooncake_session_id in self.failed_sessions:
                            self.record_failure(
                                embedding_chunk.room,
                                f"Language instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                            )
                            self.update_status(embedding_chunk.room, KVPoll.Failed)
                            self.sync_status_to_language_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                            )
                            break

                    logger.debug(
                        f"Sending embedding data for request {req.mooncake_session_id} with {embedding_chunk.embedding_index=};{self.language_args_table[req.mooncake_session_id].dst_embedding_ptrs=};{req.dst_embedding_index=}"
                    )
                    ret = self.send_embedding(
                        req.mooncake_session_id,
                        embedding_chunk.embedding_index,
                        self.language_args_table[
                            req.mooncake_session_id
                        ].dst_embedding_ptrs,
                        req.dst_embedding_index,
                        embedding_chunk.chunk_info,
                    )
                    if ret != 0:
                        with self.session_lock:
                            self.session_failures[req.mooncake_session_id] += 1
                            # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                            if self.session_failures[req.mooncake_session_id] >= 1:
                                self.failed_sessions.add(req.mooncake_session_id)
                                logger.error(
                                    f"Session {req.mooncake_session_id} failed."
                                )
                            logger.error(
                                f"Session {req.mooncake_session_id} failed with {embedding_chunk.room=};{req.endpoint=};{req.dst_port=};{req.room=}"
                            )
                        self.record_failure(
                            embedding_chunk.room,
                            f"Failed to send embedding chunk of {embedding_chunk.room} to {req.endpoint}:{req.dst_port}",
                        )
                        self.update_status(embedding_chunk.room, KVPoll.Failed)
                        self.sync_status_to_language_endpoint(
                            req.endpoint, req.dst_port, req.room, KVPoll.Failed
                        )
                        break

                    polls.append(True if ret == 0 else False)
                    dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))

                    # Only sync status when all the dst ranks have received the embedding data
                    if len(polls) == req.required_dst_info_num:
                        status = KVPoll.Success if all(polls) else KVPoll.Failed
                        logger.debug(
                            f"Transfer completed: polls={polls}, status={status}"
                        )
                        self.update_status(req.room, status)
                        for endpoint, dst_port, room in dst_ranks_infos:
                            logger.debug(
                                f"Syncing status to {endpoint}:{dst_port}, room={room}, status={status}"
                            )
                            self.sync_status_to_language_endpoint(
                                endpoint, dst_port, room, status
                            )

                if (
                    embedding_chunk.room not in self.request_status
                    or self.check_status(embedding_chunk.room) == KVPoll.Success
                ):
                    if embedding_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(embedding_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Embedding instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_embedding_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")
        logger.debug(f"Start embedding thread with {self.rank_port=}")

        def embedding_thread():
            """This thread recvs pre-alloc notification from the language engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.language_args_table[mooncake_session_id] = (
                        EmbeddingArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(
                        f"[Register]Language args table: {mooncake_session_id=} {self.language_args_table[mooncake_session_id]=}"
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[5].decode("ascii"))
                    room = int(room)
                    logger.debug(
                        f"Bootstrap thread received request with {room=};{mooncake_session_id=};{required_dst_info_num=}"
                    )
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferEmbeddingInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(f"Transfer infos: {self.transfer_infos=}")
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        logger.debug(f"Marking room {room} as waiting for input")
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=embedding_thread).start()

    def start_language_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def language_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                logger.debug(
                    f"Language thread received request with {bootstrap_room=};{status=}"
                )
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                if status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get embedding data from embedding instance, it might be dead",
                    )
                self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.embedding_dp_size_table.keys())

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

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the tracker
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
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

        threading.Thread(target=language_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        embedding_index: int,
        is_last: bool,
        chunk_info: List[Tuple[int, int]],
    ):
        assert self.disaggregation_mode == DisaggregationMode.EMBEDDING
        assert is_last  # For embedding data, we only send once at the end

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)
        logger.debug(
            f"Add transfer request to shard {shard_idx} with {bootstrap_room=};{embedding_index=};{is_last=}"
        )

        self.transfer_queues[shard_idx].put(
            TransferEmbeddingChunk(
                room=bootstrap_room,
                embedding_index=embedding_index,
                is_last=is_last,
                chunk_info=chunk_info,
            )
        )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            logger.debug(
                f"Update status for room {bootstrap_room} from None to {status}"
            )
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                logger.debug(
                    f"Update status for room {bootstrap_room} from {self.request_status[bootstrap_room]} to KVPoll.Failed"
                )
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                logger.debug(
                    f"Update status for room {bootstrap_room} from {self.request_status[bootstrap_room]} to {status}"
                )
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self):
        return self.engine.get_session_id()

    def _register_to_bootstrap(self):
        """Register EmbeddingSender to bootstrap server via HTTP POST."""
        bootstrap_server_url = f"{get_local_ip_by_remote()}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Embedding",
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.data_args.engine_rank,
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Embedding successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Embedding instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Embedding instance failed to register to bootstrap server: {e}"
            )

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.embedding_dp_size_table:
                del self.embedding_dp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
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
                    f"Losing connection with embedding instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with embedding instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )


class MooncakeEmbeddingSender(BaseKVSender):
    def __init__(
        self, mgr: MooncakeEmbeddingManager, bootstrap_addr: str, bootstrap_room: int
    ):
        self.embedding_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.embedding_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.embedding_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state = None
        self.init_time = None
        logger.debug(
            f"Init embedding sender with {bootstrap_room=};{bootstrap_addr=};{self.embedding_index=}"
        )

    def init(self, embedding_index: Optional[int] = None):
        # For embedding data, we don't need num_kv_indices, but we keep the interface consistent
        self.embedding_index = embedding_index
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
    ):
        # For embedding data, we don't use kv_indices, but we keep the interface consistent
        # We only send embedding data once at the end
        pass

    def send_embedding(self, embedding_index: int, last_chunk: bool, chunk_info: List[Tuple[int, int]]):
        """Send embedding data to language instances"""
        self.embedding_mgr.add_transfer_request(
            self.bootstrap_room, embedding_index, last_chunk, chunk_info
        )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.embedding_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.embedding_mgr.bootstrap_time_out:
                        self.embedding_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.embedding_mgr.request_status:
            self.embedding_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()

        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        with self.embedding_mgr.failure_lock:
            failure_reason = self.embedding_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise EmbeddingTransferError(self.bootstrap_room, failure_reason)


class MooncakeEmbeddingReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: MooncakeEmbeddingManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.embedding_mgr = mgr
        self.session_id = self.embedding_mgr.get_session_id()
        logger.debug(f"Receiver session_id: {self.session_id}")
        self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None
        self.data_parallel_rank = data_parallel_rank

        if self.bootstrap_addr not in self.embedding_mgr.embedding_dp_size_table:
            self.embedding_tp_size, self.embedding_dp_size = (
                self._get_embedding_parallel_info_from_server()
            )
            logger.debug(
                f"Fetch embedding parallel info from [{self.bootstrap_addr}]: DP size:{self.embedding_dp_size}, TP size:{self.embedding_tp_size}"
            )
            if self.embedding_tp_size is None or self.embedding_dp_size is None:
                self.embedding_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch embedding parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            else:
                logger.debug(
                    f"Fetch embedding parallel info from [{self.bootstrap_addr}]: DP size:{self.embedding_dp_size}; TP size:{self.embedding_tp_size}"
                )
                self.embedding_mgr.embedding_dp_size_table[self.bootstrap_addr] = (
                    self.embedding_dp_size
                )
                self.embedding_mgr.embedding_tp_size_table[self.bootstrap_addr] = (
                    self.embedding_tp_size
                )
        else:
            self.embedding_tp_size = self.embedding_mgr.embedding_tp_size_table[
                self.bootstrap_addr
            ]
            self.embedding_dp_size = self.embedding_mgr.embedding_dp_size_table[
                self.bootstrap_addr
            ]

        local_tp_size_per_dp_rank = (
            self.embedding_mgr.tp_size // self.embedding_mgr.dp_size
        )
        if local_tp_size_per_dp_rank <= self.embedding_tp_size:
            self.target_tp_rank = (
                self.embedding_mgr.data_args.engine_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            self.target_tp_rank = (
                self.embedding_mgr.data_args.engine_rank % self.embedding_tp_size
            )
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // self.embedding_tp_size
            )
            self.target_tp_ranks = [self.target_tp_rank]

        if self.data_parallel_rank is not None:
            logger.debug(f"Targeting DP rank: {self.data_parallel_rank}")
            self.target_dp_group = self.data_parallel_rank
        else:
            self.target_dp_group = bootstrap_room % self.embedding_dp_size

        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        logger.debug(f"Bootstrap key: {bootstrap_key};")

        if bootstrap_key not in self.embedding_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
                    logger.debug(
                        f"Fetched bootstrap info: {bootstrap_info} for DP {self.target_dp_group} TP {target_tp_rank}"
                    )
                    bootstrap_infos.append(bootstrap_info)
                else:
                    self.embedding_mgr.record_failure(
                        self.bootstrap_room,
                        f"Could not fetch bootstrap info for engine rank: {self.embedding_mgr.data_args.engine_rank} and target_dp_group: {self.target_dp_group}",
                    )
                    self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

            self.bootstrap_infos = bootstrap_infos
            self.embedding_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

            # Register aux_args only once to prefill AuxManager according to the info fetched from the bootstrap server
            self._register_embedding_args()
        else:
            self.bootstrap_infos = self.embedding_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0
        self.embedding_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(
            self.bootstrap_room
        )
        self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

        logger.debug(
            f"Init embedding receiver with {self.bootstrap_room=};{self.bootstrap_addr=};{self.embedding_tp_size=};{self.embedding_dp_size=};{self.target_dp_group=};{self.target_tp_rank=};{self.target_tp_ranks=};{self.required_dst_info_num=};{self.data_parallel_rank=}"
        )

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                logger.debug(
                    f"Fetched bootstrap info: {bootstrap_info} for engine rank: {engine_rank} and target_dp_group: {target_dp_group}"
                )
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _get_embedding_parallel_info_from_server(self) -> Tuple[int, int]:
        """Fetch the embedding parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                embedding_parallel_info = response.json()
                return int(embedding_parallel_info["embedding_tp_size"]), int(
                    embedding_parallel_info["embedding_dp_size"]
                )
            else:
                logger.error(
                    f"Failed to get embedding parallel info: {response.status_code}, {response.text}"
                )
                return None, None
        except Exception as e:
            logger.error(f"Error fetching embedding parallel info from bootstrap: {e}")
            return None, None

    def _register_embedding_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.debug(f"Register embedding args to [{self.embedding_server_url}]")
            packed_embedding_data_ptrs = b"".join(
                struct.pack("Q", ptr)
                for ptr in self.embedding_mgr.data_args.aux_data_ptrs
            )

            sock, lock = self._connect("tcp://" + self.embedding_server_url)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.embedding_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_embedding_data_ptrs,
                    ]
                )
                logger.debug(
                    f"Register embedding args with {self.embedding_mgr.rank_port=};{self.session_id=};{self.embedding_mgr.data_args.aux_data_ptrs=} success"
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

    def init(self, embedding_index: Optional[int] = None):
        for bootstrap_info in self.bootstrap_infos:
            self.embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )

            sock, lock = self._connect("tcp://" + self.embedding_server_url)
            logger.debug(
                f"Init embedding receiver with {self.bootstrap_room=};{self.embedding_server_url=};{self.embedding_mgr.rank_port=};{self.session_id=};{embedding_index=};{self.required_dst_info_num=}"
            )
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.embedding_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        str(embedding_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
            logger.debug(
                f"Init embedding receiver with {self.bootstrap_room=};{self.embedding_server_url=};{self.embedding_mgr.rank_port=};{self.session_id=};{embedding_index=};{self.required_dst_info_num=} success"
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.embedding_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.embedding_mgr.request_status:
            self.embedding_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()

        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        with self.embedding_mgr.failure_lock:
            failure_reason = self.embedding_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise EmbeddingTransferError(self.bootstrap_room, failure_reason)


class MooncakeEmbeddingBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.tp_size = None
        self.dp_size = None
        self.embedding_port_table: Dict[int, Dict[int, Dict[str, Union[str, int]]]] = {}

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
        logger.debug(f"Register embedding bootstrap: {request}")
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

        if role == "Embedding":
            dp_group = engine_rank // self.tp_size
            tp_rank_in_dp_group = engine_rank % self.tp_size

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.embedding_port_table:
                    self.embedding_port_table[dp_group] = {}

            self.embedding_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register embedding bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            embedding_parallel_info = {
                "embedding_tp_size": self.tp_size,
                "embedding_dp_size": self.dp_size,
            }
            return web.json_response(embedding_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.embedding_port_table[int(target_dp_group)][
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
