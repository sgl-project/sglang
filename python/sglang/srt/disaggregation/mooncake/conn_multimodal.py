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
from sglang.srt.disaggregation.common.utils import FastQueue
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_free_port, get_int_env_var, get_local_ip_by_remote

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
    embedding_indices: List[int]  # Source embedding indices
    is_last: bool
    total_tokens: int  # Total number of tokens to transfer


@dataclasses.dataclass
class TransferEmbeddingInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_indices: List[int]
    required_dst_info_num: int
    sent_tokens: int = 0  # Number of tokens already sent (for resume transfer)
    allocated_tokens: int = 0  # Number of tokens allocated by Language side
    # For resume: need to store original embedding data to retrigger transfer
    src_embedding_indices: List[int] = (
        None  # Source embedding indices (from Embedding side)
    )
    total_tokens: int = 0  # Total tokens to transfer (from Embedding side)
    resume_ready: bool = False  # Whether ready for resume transfer

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Parse embedding_indices from message
        # Format: comma-separated list of embedding indices
        dst_embedding_indices_str = msg[4].decode("ascii")
        if dst_embedding_indices_str:
            dst_embedding_indices = [
                int(x) for x in dst_embedding_indices_str.split(",")
            ]
        else:
            dst_embedding_indices = []

        required_dst_info_num = int(msg[5].decode("ascii"))

        # Parse allocated_tokens (always present in init message, msg[6])
        # For resume messages, msg[6] is sent_tokens, msg[7] is allocated_tokens
        allocated_tokens = 0
        sent_tokens = 0

        if len(msg) >= 7:
            # Check if this is a resume message (has 8 fields) or init message (has 7 fields)
            if len(msg) >= 8:
                # Resume message: msg[6] = sent_tokens, msg[7] = allocated_tokens
                sent_tokens = int(msg[6].decode("ascii"))
                allocated_tokens = int(msg[7].decode("ascii"))
            else:
                # Init message: msg[6] = allocated_tokens
                allocated_tokens = int(msg[6].decode("ascii"))

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_indices=dst_embedding_indices,
            required_dst_info_num=required_dst_info_num,
            sent_tokens=sent_tokens,
            allocated_tokens=allocated_tokens,
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

        if self.disaggregation_mode == DisaggregationMode.ENCODE:
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
        embedding_indices: List[int],
        dst_embedding_ptrs: list[int],
        dst_embedding_indices: List[int],
        total_tokens: int,
        block_size: int,
        sent_tokens: int = 0,
        allocated_tokens: int = None,
    ):
        """Send embedding data using block-based transfer.

        Args:
            mooncake_session_id: Session ID for transfer
            embedding_indices: Source embedding indices
            dst_embedding_ptrs: Destination buffer pointers
            dst_embedding_indices: Destination embedding indices
            total_tokens: Total number of tokens to transfer
            block_size: Number of tokens per block
            sent_tokens: Number of tokens already sent (for resume transfer)
            allocated_tokens: Number of tokens allocated by Language side

        Returns:
            Tuple of (ret, is_partial):
                ret: 0 if all transfers succeed, 1 otherwise
                is_partial: True if this is a partial transfer (more data remaining)
        """
        # Validate block_size consistency
        if allocated_tokens is not None:
            expected_block_size = allocated_tokens // len(dst_embedding_indices)
            if expected_block_size != block_size:
                raise ValueError(
                    f"Block size mismatch: Embedding side uses {block_size}, "
                    f"but Language side allocated {allocated_tokens} tokens "
                    f"for {len(dst_embedding_indices)} blocks "
                    f"(implies block_size={expected_block_size})"
                )
        else:
            # Backward compatibility: if no allocated_tokens, calculate from block count
            allocated_tokens = len(dst_embedding_indices) * block_size

        # Calculate remaining tokens and determine if this is a partial transfer
        remaining_tokens = total_tokens - sent_tokens

        if remaining_tokens > allocated_tokens:
            # Need partial transfer
            logger.info(
                f"Partial transfer: remaining={remaining_tokens} > "
                f"allocated={allocated_tokens}. Will transfer {allocated_tokens} tokens."
            )
            tokens_to_send = allocated_tokens
            is_partial = True
        else:
            # Can transfer all remaining tokens
            tokens_to_send = remaining_tokens
            is_partial = False

        # Calculate required dst blocks
        dst_blocks_needed = (tokens_to_send + block_size - 1) // block_size

        # Validate dst buffer is sufficient
        if dst_blocks_needed > len(dst_embedding_indices):
            raise ValueError(
                f"Insufficient dst blocks: need {dst_blocks_needed} blocks "
                f"for {tokens_to_send} tokens, but only have {len(dst_embedding_indices)} blocks"
            )

        # Calculate source block range based on sent_tokens
        start_block = sent_tokens // block_size
        embedding_indices_to_send = embedding_indices[
            start_block : start_block + dst_blocks_needed
        ]
        dst_embedding_indices = dst_embedding_indices[:dst_blocks_needed]

        src_addrs = []
        dst_addrs = []
        lengths = []

        tokens_transferred = 0

        for block_idx, (src_block_idx, dst_block_idx) in enumerate(
            zip(embedding_indices_to_send, dst_embedding_indices)
        ):
            # Calculate tokens in this block
            remaining_in_transfer = tokens_to_send - tokens_transferred
            tokens_in_block = min(block_size, remaining_in_transfer)

            if tokens_in_block <= 0:
                break

            # Transfer each buffer type within the block
            for buffer_type_idx in range(len(self.data_args.aux_item_lens)):
                embedding_item_len = self.data_args.aux_item_lens[buffer_type_idx]

                # Calculate chunk size based on buffer type and tokens_in_block
                # For aux_datas, only transfer in first block of initial transfer
                if buffer_type_idx == 3:  # aux_datas
                    if sent_tokens == 0 and block_idx == 0:
                        chunk_size = embedding_item_len  # Transfer full aux_datas
                    else:
                        continue  # Skip aux_datas for resume or other blocks
                else:
                    # For embeddings, fill_ids, mrope_positions: scale by tokens_in_block
                    # embedding_item_len already contains the full block size
                    # We need to transfer only tokens_in_block portion
                    chunk_size = (embedding_item_len * tokens_in_block) // block_size

                # Calculate source address: base_ptr + src_block_idx * item_len
                embedding_addr = (
                    self.data_args.aux_data_ptrs[buffer_type_idx]
                    + src_block_idx * embedding_item_len
                )

                # Calculate destination address: base_ptr + dst_block_idx * item_len
                dst_embedding_addr = (
                    dst_embedding_ptrs[buffer_type_idx]
                    + dst_block_idx * embedding_item_len
                )

                src_addrs.append(embedding_addr)
                dst_addrs.append(dst_embedding_addr)
                lengths.append(chunk_size)

            tokens_transferred += tokens_in_block

        ret = self.engine.batch_transfer_sync(
            mooncake_session_id, src_addrs, dst_addrs, lengths
        )

        return ret, is_partial

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

                for req in reqs_to_be_processed:
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

                    # Save source embedding info for potential resume
                    if req.src_embedding_indices is None:
                        req.src_embedding_indices = embedding_chunk.embedding_indices
                        req.total_tokens = embedding_chunk.total_tokens

                    # Block-based transfer
                    # Calculate block_size from aux_item_lens
                    # aux_item_lens[0] is for embeddings per block
                    # aux_item_lens[1] is for fill_ids per block
                    # block_size = aux_item_lens[1] / fill_ids.itemsize
                    # Assuming fill_ids is int32 (4 bytes)
                    block_size = self.data_args.aux_item_lens[1] // 4

                    # Get sent_tokens and allocated_tokens from transfer_info
                    sent_tokens = req.sent_tokens
                    allocated_tokens = req.allocated_tokens

                    ret, is_partial = self.send_embedding(
                        req.mooncake_session_id,
                        embedding_chunk.embedding_indices,
                        self.language_args_table[
                            req.mooncake_session_id
                        ].dst_embedding_ptrs,
                        req.dst_embedding_indices,
                        embedding_chunk.total_tokens,
                        block_size,
                        sent_tokens,
                        allocated_tokens,
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

                    # Update sent_tokens after successful transfer
                    tokens_sent = min(
                        embedding_chunk.total_tokens - sent_tokens, allocated_tokens
                    )
                    req.sent_tokens += tokens_sent

                    polls.append(True if ret == 0 else False)
                    dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))

                    # Only sync status when all the dst ranks have received the embedding data
                    if len(polls) == req.required_dst_info_num:
                        if is_partial:
                            # Partial transfer complete, waiting for resume
                            status = (
                                KVPoll.Transferring if all(polls) else KVPoll.Failed
                            )
                        else:
                            # Complete transfer done
                            status = KVPoll.Success if all(polls) else KVPoll.Failed

                        self.update_status(req.room, status)
                        for endpoint, dst_port, room in dst_ranks_infos:
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
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    continue
                else:
                    room = int(room)

                    # Check if this is a resume request (8 fields) or init request (7 fields)
                    is_resume = len(waiting_req_bytes) >= 8

                    if is_resume:
                        # Resume request: update existing transfer_info and trigger transfer
                        if (
                            room in self.transfer_infos
                            and mooncake_session_id in self.transfer_infos[room]
                        ):
                            transfer_info = TransferEmbeddingInfo.from_zmq(
                                waiting_req_bytes
                            )
                            req = self.transfer_infos[room][mooncake_session_id]

                            # Update the existing transfer_info with resume data
                            req.sent_tokens = transfer_info.sent_tokens
                            req.allocated_tokens = transfer_info.allocated_tokens
                            req.dst_embedding_indices = (
                                transfer_info.dst_embedding_indices
                            )

                            logger.debug(
                                f"Resume transfer for room={room}, sent_tokens={transfer_info.sent_tokens}, "
                                f"sent_tokens={transfer_info.sent_tokens}, "
                                f"allocated_tokens={transfer_info.allocated_tokens}"
                            )

                            # Don't reset status - it should remain in current state (Transferring)

                            req.resume_ready = True
                            # Check if all sessions are ready for resume (similar to init logic)
                            all_dst_ranks_ready = all(
                                dst_req.resume_ready
                                for dst_req in self.transfer_infos[room].values()
                            )

                            # Only trigger resume transfer when all dst ranks are ready
                            if all_dst_ranks_ready:
                                if (
                                    req.src_embedding_indices is not None
                                    and req.total_tokens > 0
                                ):
                                    # Calculate which queue to use (same as add_transfer_request)
                                    dst_infos = self.transfer_infos[room].keys()
                                    session_port_sum = sum(
                                        int(session.split(":")[1])
                                        for session in dst_infos
                                    )
                                    shard_idx = session_port_sum % len(
                                        self.transfer_queues
                                    )

                                    # Add resume transfer chunk to queue (only once for all sessions)
                                    self.transfer_queues[shard_idx].put(
                                        TransferEmbeddingChunk(
                                            room=room,
                                            embedding_indices=req.src_embedding_indices,
                                            is_last=True,  # Resume is always the last part
                                            total_tokens=req.total_tokens,
                                        )
                                    )

                                    logger.debug(
                                        f"Resume transfer triggered: room={room}, "
                                        f"queue_idx={shard_idx}, src_blocks={len(req.src_embedding_indices)}, "
                                        f"all {len(self.transfer_infos[room])} sessions ready"
                                    )
                                    for dst_req in self.transfer_infos[room].values():
                                        dst_req.resume_ready = (
                                            False  # Reset for potential future resumes
                                        )
                                else:
                                    logger.error(
                                        f"Cannot trigger resume: missing src_embedding_indices or total_tokens "
                                        f"for room={room} session={mooncake_session_id}"
                                    )
                            else:
                                logger.debug(
                                    f"Waiting for all dst ranks to be ready for resume: room={room}, "
                                    f"ready={sum(dst_req.resume_ready for dst_req in self.transfer_infos[room].values())}/{len(self.transfer_infos[room])}"
                                )
                        else:
                            logger.error(
                                f"Cannot resume: room={room} session={mooncake_session_id} not found in transfer_infos"
                            )
                    else:
                        # Init request: create new transfer_info
                        required_dst_info_num = int(
                            waiting_req_bytes[5].decode("ascii")
                        )

                        if room not in self.transfer_infos:
                            self.transfer_infos[room] = {}

                        self.transfer_infos[room][mooncake_session_id] = (
                            TransferEmbeddingInfo.from_zmq(waiting_req_bytes)
                        )
                        # NOTE: after bootstrapping we can mark the req as waiting for input
                        if len(self.transfer_infos[room]) == required_dst_info_num:
                            self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=embedding_thread).start()

    def start_language_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def language_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
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
        embedding_indices: List[int],
        is_last: bool,
        total_tokens: int,
        block_size: int,
    ):
        """Add block-based transfer request to queue.

        Note:
            - This method is only called for initial transfer (by send_embedding)
            - Resume transfers are triggered by Language side sending resume messages
            - sent_tokens information is maintained in transfer_infos

        Args:
            bootstrap_room: Room ID for the request
            embedding_indices: List of source embedding indices
            is_last: Whether this is the last chunk
            total_tokens: Total number of tokens to transfer
            block_size: Number of tokens per block
        """
        assert self.disaggregation_mode == DisaggregationMode.ENCODE
        assert is_last  # For embedding data, we only send once at the end

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # Prevent duplicate transfer: if already in Transferring or Success state, skip
        current_status = self.check_status(bootstrap_room)
        if current_status in [KVPoll.Transferring, KVPoll.Success]:
            logger.debug(
                f"Skip duplicate transfer for room={bootstrap_room}, status={current_status}"
            )
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferEmbeddingChunk(
                room=bootstrap_room,
                embedding_indices=embedding_indices,
                is_last=is_last,
                total_tokens=total_tokens,
            )
        )

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
        """Register EmbeddingSender to bootstrap server via HTTP POST."""
        bootstrap_server_url = f"{get_local_ip_by_remote()}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Encode",
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
        self,
        mgr: MooncakeEmbeddingManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.embedding_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.embedding_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.embedding_indices = None
        self.bootstrap_server_url = bootstrap_addr
        self.conclude_state = None
        self.init_time = None
        self.dest_tp_ranks = dest_tp_ranks
        self.pp_rank = pp_rank

    def init(
        self,
        embedding_indices: Optional[List[int]] = None,
    ):
        """Initialize sender with embedding indices.

        Args:
            embedding_indices: List of embedding indices for block-based allocation
        """
        self.embedding_indices = embedding_indices
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
    ):
        # For embedding data, we don't use kv_indices, but we keep the interface consistent
        # We only send embedding data once at the end
        pass

    def send_embedding(
        self,
        embedding_indices: List[int] = None,
        last_chunk: bool = True,
        total_tokens: int = None,
        block_size: int = None,
    ):
        """Send embedding data to language instances using block-based transfer.

        Args:
            embedding_indices: List of source embedding indices
            last_chunk: Whether this is the last chunk
            total_tokens: Total number of tokens to transfer
            block_size: Number of tokens per block
        """
        self.embedding_mgr.add_transfer_request(
            self.bootstrap_room, embedding_indices, last_chunk, total_tokens, block_size
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

        if self.bootstrap_room in self.embedding_mgr.transfer_infos:
            self.embedding_mgr.transfer_infos.pop(self.bootstrap_room)

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
        prefill_dp_rank: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.embedding_mgr = mgr
        self.session_id = self.embedding_mgr.get_session_id()
        self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.conclude_state = None
        self.data_parallel_rank = prefill_dp_rank

        if self.bootstrap_addr not in self.embedding_mgr.embedding_dp_size_table:
            self.embedding_tp_size, self.embedding_dp_size = (
                self._get_embedding_parallel_info_from_server()
            )
            if self.embedding_tp_size is None or self.embedding_dp_size is None:
                self.embedding_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch embedding parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.embedding_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            else:
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
            self.target_dp_group = self.data_parallel_rank
        else:
            self.target_dp_group = bootstrap_room % self.embedding_dp_size

        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.embedding_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
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

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    def init(
        self,
        embedding_indices: Optional[List[int]] = None,
        allocated_tokens: Optional[int] = None,
    ):
        """Initialize receiver with embedding indices.

        Args:
            embedding_indices: List of embedding indices for block-based allocation
            allocated_tokens: Number of tokens allocated by Language side
        """
        if embedding_indices is not None:
            embedding_indices_str = ",".join(str(idx) for idx in embedding_indices)
        else:
            embedding_indices_str = ""

        # Calculate allocated_tokens if not provided
        if allocated_tokens is None and embedding_indices is not None:
            # Calculate from block indices
            # block_size = aux_item_lens[1] / fill_ids.itemsize (assuming int32 = 4 bytes)
            block_size = self.embedding_mgr.data_args.aux_item_lens[1] // 4
            allocated_tokens = len(embedding_indices) * block_size

        for bootstrap_info in self.bootstrap_infos:
            self.embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )

            sock, lock = self._connect("tcp://" + self.embedding_server_url)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.embedding_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        embedding_indices_str.encode(
                            "ascii"
                        ),  # Send comma-separated embedding indices
                        str(self.required_dst_info_num).encode("ascii"),
                        str(allocated_tokens).encode("ascii"),  # Send allocated tokens
                    ]
                )

    def resume_transfer(
        self,
        embedding_indices: List[int],
        sent_tokens: int,
        allocated_tokens: int,
    ):
        """Resume transfer with new allocation after partial transfer.

        Args:
            embedding_indices: New block allocation for remaining data
            sent_tokens: Number of tokens already transferred
            allocated_tokens: Number of tokens in new allocation
        """
        embedding_indices_str = ",".join(str(idx) for idx in embedding_indices)

        for bootstrap_info in self.bootstrap_infos:
            self.embedding_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )

            sock, lock = self._connect("tcp://" + self.embedding_server_url)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.embedding_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        embedding_indices_str.encode("ascii"),  # New allocation
                        str(self.required_dst_info_num).encode("ascii"),
                        str(sent_tokens).encode("ascii"),  # Resume marker
                        str(allocated_tokens).encode("ascii"),  # New allocation size
                    ]
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

    def abort(self):
        self.embedding_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeEmbeddingBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, host: str, port: int):
        self.port = port
        self.host = host
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

        if role == "Encode":
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
