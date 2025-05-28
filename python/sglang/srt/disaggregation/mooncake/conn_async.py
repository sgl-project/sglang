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
from collections import deque
from functools import cache, reduce
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
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
    TransferKVChunk,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FakeBootstrapHost,
    StreamAsyncSubmitter,
    get_src_dst_index_length,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_free_port, get_ip, get_local_ip_by_remote

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TransferKVChunkSet:
    rooms: Tuple[int] = dataclasses.field(default_factory=tuple)
    prefill_kv_indices: Tuple[npt.NDArray[np.int64]] = dataclasses.field(
        default_factory=tuple
    )
    index_slices: Tuple[slice] = dataclasses.field(default_factory=tuple)


@dataclasses.dataclass
class AsyncInfo:
    layer_ids: Tuple[int] = dataclasses.field(default_factory=tuple)
    kv_chunk_info: TransferKVChunkSet = dataclasses.field(
        default_factory=TransferKVChunkSet
    )


class MooncakeAsyncKVManager(MooncakeKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        # [TODO] support other backends
        if not is_mla_backend:
            logger.error("MooncakeAsyncKVManager is not supported for other backend.")
            raise NotImplementedError(
                "MooncakeAsyncKVManager is not supported for other backend."
            )
        self._sem = threading.Semaphore(0)
        submit_func = lambda: self._sem.release()
        self._async_submitter = StreamAsyncSubmitter(submit_func)
        self._notify_queue = deque()
        self._init_put_kvcache_thread()
        self._waiting_rooms = deque()
        self._current_kv_chunk_infos: Optional[TransferKVChunkSet] = None
        self._req_begin_count: Dict[int, int] = {}
        self._req_bids: Dict[int, List[Tuple[int]]] = {}
        self._lock = threading.Lock()

    @property
    def is_support_asnyc(self):
        return True

    def _put_kvcache_thread_func(self):
        while True:
            self._sem.acquire()
            try:
                info = self._notify_queue.pop()
                self._put_kv_cache_internal(info)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error in put_kvcache_thread: {e}", flush=True)
                import os

                os._exit(1)

    def _init_put_kvcache_thread(self):
        self._put_kvcache_thread = threading.Thread(
            target=self._put_kvcache_thread_func, daemon=True
        )
        self._put_kvcache_thread.start()
        print(f"start put_kvcache_thread : {self._put_kvcache_thread}")

    def get_info_with_risk(self, room: int) -> TransferInfo:
        """
        there is no lock to protect the self.transfer_infos, so it have the risk, but it is ok because we delete the task after the transfer is done.
        """
        return self.transfer_infos[room]

    # Worker function for processing a single layer
    def submit_layer(
        self,
        session_id: str,
        src_ptr: int,
        dst_ptr: int,
        prefill_kv_blocks: npt.NDArray[np.int64],
        dst_kv_blocks: npt.NDArray[np.int64],
        item_len: int,
    ) -> int:
        prefill_kv_blocks, dst_kv_blocks, block_lengths = get_src_dst_index_length(
            tuple(prefill_kv_blocks), tuple(dst_kv_blocks)
        )
        # [TODO] support the blocks == 1 for now
        assert len(prefill_kv_blocks) == len(dst_kv_blocks)
        bids = []
        for prefill_index, decode_index, block_length in zip(
            prefill_kv_blocks, dst_kv_blocks, block_lengths
        ):
            src_addr = src_ptr + int(prefill_index) * item_len
            dst_addr = dst_ptr + int(decode_index) * item_len
            length = item_len * block_length
            batch_id = self.engine.transfer_submit_write(
                session_id, src_addr, dst_addr, length
            )
            bids.append(batch_id)
        return tuple(bids)

    def _put_kv_cache_internal(self, async_info: AsyncInfo):
        kv_chunk_info = async_info.kv_chunk_info
        infos = [self.get_info_with_risk(room) for room in kv_chunk_info.rooms]
        for layer_id in async_info.layer_ids:
            for room_id, transfer_info_dict, kv_indice, index_slice in zip(
                kv_chunk_info.rooms,
                infos,
                kv_chunk_info.prefill_kv_indices,
                kv_chunk_info.index_slices,
            ):
                for transfer_info in transfer_info_dict.values():
                    if not transfer_info.is_dummy:
                        src = kv_indice
                        # the src
                        dst = transfer_info.dst_kv_indices[index_slice]
                        session_id = transfer_info.mooncake_session_id
                        dst_kv_ptrs = self.decode_kv_args_table[
                            transfer_info.mooncake_session_id
                        ].dst_kv_ptrs
                        src_ptr = self.kv_args.kv_data_ptrs[layer_id]
                        dst_ptr = dst_kv_ptrs[layer_id]
                        item_len = self.kv_args.kv_item_lens[layer_id]
                        bids = self.submit_layer(
                            session_id, src_ptr, dst_ptr, src, dst, item_len
                        )
                        with self._lock:
                            if room_id not in self._req_bids:
                                self._req_bids[room_id] = []
                            self._req_bids[room_id].append(bids)

    def mark_layer_ready(self, layer_id: int):
        # for first layer, we need update the kv_chunk_info
        if layer_id == 0:
            begin_count = self._async_submitter.get_step_count()
            self._current_kv_chunk_infos = self._waiting_rooms.pop()
            # and we need to update the req_finish_count
            if self._current_kv_chunk_infos:
                for rid in self._current_kv_chunk_infos.rooms:
                    self._req_begin_count[rid] = begin_count
        # and then, we need to push the info to queue, and do step_async
        if self._current_kv_chunk_infos:
            self._notify_queue.appendleft(
                AsyncInfo(
                    layer_ids=(layer_id,), kv_chunk_info=self._current_kv_chunk_infos
                )
            )
            self._async_submitter.step_async()

    def _is_bids_finished_func(self, rid: int):
        num_layers = len(self.kv_args.kv_data_ptrs)
        is_bids_finished = (
            rid not in self._req_bids or len(self._req_bids[rid]) != num_layers
        )
        return is_bids_finished

    def pop_req_bids(self, rid: int):
        return self._req_bids.pop(rid)

    def _flush_all_layers(self, rid: int):
        start_time = time.time()
        # we have to make sure the send is finished, to do so, we need to get sent count and wait for it
        current_sent_count = self._async_submitter.get_sent_count()
        begin_count = self._req_begin_count[rid]
        num_layers = len(self.kv_args.kv_data_ptrs)
        self._async_submitter.wait_sent_finish(begin_count + num_layers)
        # and we have to make sure all the layers are sent out

        while self._is_bids_finished_func(rid):
            # print(f"self._req_bids[{rid}]({len(self._req_bids[rid])}) == {self._req_bids[rid]}")
            time.sleep(1e-3)
        # print(f"all layers are sent out of {rid}, we need to wait it finish")

        bids = reduce(lambda x, y: list(x) + list(y), self.pop_req_bids(rid), [])
        for bid in bids:
            status = self.engine.transfer_check_status(bid)
            while status == 0:
                status = self.engine.transfer_check_status(bid)
                time.sleep(1e-3)
            assert status == 1, f"status is {status} in {rid}"
        print(
            f"finish send (rid={rid}, n_blocks={bids}) in {1000*(time.time() - start_time)} ms."
        )

    def prepare_batch(self, sch: "Scheduler", batch: "ScheduleBatch"):
        # we have to prepare the batch for each forward
        rooms = []
        kv_chunk_info_set = None
        prefill_kv_indices = []
        index_slices = []
        for req in batch.reqs:
            if req.bootstrap_host == FakeBootstrapHost:
                continue
            # print(f"req={req}, start_idx = {req.start_send_idx}, fill_ids = {req.fill_ids}, origin_input_ids={req.origin_input_ids}")
            kv_chunk_info: Tuple[Tuple[npt.NDArray[np.int64], slice], int] = (
                sch.get_kv_chunk_info(req, delay_send=False)
            )
            if kv_chunk_info is not None:
                (page_indices, indexs), _ = kv_chunk_info
                bootstrap_room = req.bootstrap_room
                rooms.append(bootstrap_room)
                prefill_kv_indices.append(page_indices)
                index_slices.append(indexs)
        # we need convert async_infos to TransferKVChunkSet
        if len(rooms):
            kv_chunk_info_set = TransferKVChunkSet(
                rooms=tuple(rooms),
                prefill_kv_indices=tuple(prefill_kv_indices),
                index_slices=tuple(index_slices),
            )
        # print(f"kv_chunk_info_set is {kv_chunk_info_set}")
        self._waiting_rooms.appendleft(kv_chunk_info_set)

    def start_prefill_thread(self):
        # copy from srt/disaggregation/mooncake/conn.py
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
                    # print(f"room[{room}] is set, transfer_infos is {self.transfer_infos[room]}")
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

                            self._flush_all_layers(kv_chunk.room)

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


class MooncakeAsyncKVSender(MooncakeKVSender):
    pass


class MooncakeAsyncKVReceiver(MooncakeKVReceiver):
    pass
