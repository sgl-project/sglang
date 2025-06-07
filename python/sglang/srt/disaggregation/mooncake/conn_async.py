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
    FastQueue,
    cached_group_concurrent_contiguous,
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
        #because the put_kvcache_func is very fast, there is no need to setup a new thread to run it.
        submit_func = lambda: self._put_kvcache_func()
        self._async_submitter = StreamAsyncSubmitter(submit_func)
        self._notify_queue = deque()
        self._waiting_rooms = deque()
        self._current_kv_chunk_infos: Optional[TransferKVChunkSet] = None
        self._req_begin_count: Dict[int, deque] = {}
        self._req_bids: Dict[int, List[Tuple[int]]] = {}
        self._lock = threading.Lock()
        self._kv_cache_ntensors = len(self.kv_args.kv_data_ptrs)
        self._kv_cache_nlayers = len(self.kv_args.kv_data_ptrs) if is_mla_backend else len(self.kv_args.kv_data_ptrs) // 2
    @property
    def is_support_asnyc(self):
        return True
    
    def _put_kvcache_func(self):
        try:
            info = self._notify_queue.pop()
            self._put_kv_cache_internal(info)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.info(f"Error in put_kvcache_thread: {e}")
            import os

            os._exit(1)


    def get_info_with_risk(self, room : int)-> TransferInfo:
        """
        there is no lock to protect the self.transfer_infos, so it have the risk, but it is ok because we delete the task after the transfer is done.
        """
        while room not in self.transfer_infos:
            # [TODO] why the room is not in transfer_infos?
            time.sleep(1e-3)
            logger.info(f"room {room} is not in transfer_infos")
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
        prefill_kv_blocks_tmp, dst_kv_blocks_tmp = cached_group_concurrent_contiguous(prefill_kv_blocks, dst_kv_blocks)
        prefill_kv_blocks = [x[0] for x in prefill_kv_blocks_tmp]
        dst_kv_blocks = [x[0] for x in dst_kv_blocks_tmp]
        block_lengths = [len(x) for x in prefill_kv_blocks_tmp]
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
                                self._req_bids[room_id] = deque()
                            self._req_bids[room_id].appendleft(bids)

    def mark_layer_ready(self, layer_id: int):
        # for first layer, we need update the kv_chunk_info
        if layer_id == 0:
            begin_count = self._async_submitter.get_step_count()
            self._current_kv_chunk_infos = self._waiting_rooms.pop()
            # and we need to update the req_finish_count
            if self._current_kv_chunk_infos:
                for rid in self._current_kv_chunk_infos.rooms:
                    if rid not in self._req_begin_count:
                        self._req_begin_count[rid] = deque()
                    self._req_begin_count[rid].appendleft(begin_count)
        # and then, we need to push the info to queue, and do step_async
        if self._current_kv_chunk_infos:
            send_layers = [layer_id,]
            if not self.is_mla_backend:
                # kv_data_ptrs = [
                #     self.get_key_buffer(i).data_ptr() for i in range(self.layer_num)
                # ] + [self.get_value_buffer(i).data_ptr() for i in range(self.layer_num)]
                send_layers.append(layer_id + self._kv_cache_nlayers)
            self._notify_queue.appendleft(AsyncInfo(layer_ids=tuple(send_layers), kv_chunk_info=self._current_kv_chunk_infos))
            self._async_submitter.step_async()

    def _is_bids_finished_func(self, rid : int):
        bids_not_finished = rid not in self._req_bids or len(self._req_bids[rid]) < self._kv_cache_ntensors
        return not bids_not_finished

    def pop_req_bids(self, rid : int, is_remove : bool):
        if is_remove:
            q = self._req_bids.pop(rid)
            assert(len(q) == self._kv_cache_ntensors)
            rsts = []
            for _ in range(self._kv_cache_ntensors):
                rsts.append(q.pop())
            return rsts
        else:
            rsts = []
            for _ in range(self._kv_cache_ntensors):
                rsts.append(self._req_bids[rid].pop())
            return rsts
        
    def _flush_all_layers(self, rid : int, is_last : bool):
        start_time = time.time()
        # we only flush the transfer when the whole prefill is all finished
        if is_last:
            # we have to make sure the send is finished, to do so, we need to get sent count and wait for it
            while len(self._req_begin_count[rid]):
                begin_count = self._req_begin_count[rid].pop()
                self._async_submitter.wait_sent_finish(begin_count + self._kv_cache_nlayers)
                # and we have to make sure all the layers are sent out
                while not self._is_bids_finished_func(rid):
                    # print(f"self._req_bids[{rid}]({len(self._req_bids[rid])}) == {self._req_bids[rid]}")
                    time.sleep(1e-3)
                current_last = len(self._req_begin_count[rid]) == 0
                bids = reduce(lambda x, y: list(x) + list(y), self.pop_req_bids(rid, current_last), [])
                for bid in bids:
                    status = self.engine.transfer_check_status(bid)
                    while status == 0:
                        status = self.engine.transfer_check_status(bid)
                        time.sleep(1e-3)
                    assert status == 1, f"status is {status} in {rid}"
            #and we need to release reqs
            self._req_begin_count.pop(rid)
            logger.info(f"finish send (rid={rid}, n_blocks={len(bids)}) in {1000*(time.time() - start_time)} ms.")

    def prepare_batch(self, sch : "Scheduler", batch : "ScheduleBatch"):
        # we have to prepare the batch for each forward
        rooms = []
        kv_chunk_info_set = None
        prefill_kv_indices = []
        index_slices = []
        for req in batch.reqs:
            if req.bootstrap_host == FakeBootstrapHost:
                continue
            # logger.info(f"req={req}, start_idx = {req.start_send_idx}, fill_ids = {req.fill_ids}, origin_input_ids={req.origin_input_ids}")
            kv_chunk_info : Tuple[npt.NDArray[np.int64], slice, bool] = sch.get_kv_chunk_info(req, delay_send = False)
            if kv_chunk_info is not None:
                (page_indices, indexs),_ = kv_chunk_info
                bootstrap_room = req.bootstrap_room
                rooms.append(bootstrap_room)
                prefill_kv_indices.append(page_indices)
                index_slices.append(indexs)
        #we need convert async_infos to TransferKVChunkSet
        if len(rooms):
            kv_chunk_info_set = TransferKVChunkSet(
                rooms=tuple(rooms),
                prefill_kv_indices=tuple(prefill_kv_indices),
                index_slices=tuple(index_slices),
            )
        # logger.info(f"kv_chunk_info_set is {kv_chunk_info_set}")
        self._waiting_rooms.appendleft(kv_chunk_info_set)
    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        # TODO: Shall we use KVPoll.Transferring state?
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
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
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                )
                                break
                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]
                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                        
                        self._flush_all_layers(kv_chunk.room, kv_chunk.is_last)

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
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

class MooncakeAsyncKVSender(MooncakeKVSender):
    pass

class MooncakeAsyncKVReceiver(MooncakeKVReceiver):
    pass