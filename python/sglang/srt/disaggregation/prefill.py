"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Infight Queue

3. Infight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.disaggregation.conn import KVArgs, KVManager, KVPoll, KVSender
from sglang.srt.disaggregation.utils import (
    ReqToMetadataIdxAllocator,
    poll_and_all_reduce,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: List[torch.Tensor],
        aux_dtype: torch.dtype,
        tp_rank: int,
        tp_size: int,
        server_args: ServerArgs, 
        gloo_group: ProcessGroup,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.aux_dtype = aux_dtype

        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.server_args = server_args
        self.bootstrap_port = server_args.disaggregation_bootstrap_port + tp_rank
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.kv_manager = self._init_kv_manager()

    def allocate_token_id(self, idx: int, token_id: int):
        assert token_id >= 0, f"token_id: {token_id} is negative"
        output_id_buffer = self.metadata_buffers[0]
        output_id_buffer[idx] = token_id

    def _init_kv_manager(self) -> KVManager:
        kv_args = KVArgs()
        kv_args.engine_rank = self.tp_rank
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens

        # Define req -> input ids buffer
        kv_args.aux_data_ptrs = [
            metadata_buffer.data_ptr() for metadata_buffer in self.metadata_buffers
        ]
        kv_args.aux_data_lens = [
            metadata_buffer.nbytes for metadata_buffer in self.metadata_buffers
        ]
        kv_args.aux_item_lens = [
            metadata_buffer[0].nbytes for metadata_buffer in self.metadata_buffers
        ]
        kv_args.ib_device = "mock"
        kv_manager = KVManager(kv_args, "prefill", bootstrap_port = self.bootstrap_port, decode_instance_number=self.server_args.disaggregation_decode_instance_num)
        return kv_manager

    def add(self, req: Req) -> None:
        assert req.bootstrap_room

        logger.info(f"[NIXL PD disagg] Bqueue enqueue. prefill_host: {req.prefill_host}, bootstrap_room: {req.bootstrap_room}")
        req.disagg_kv_sender = KVSender(
            mgr=self.kv_manager, 
            bootstrap_room=req.bootstrap_room,
        )
        self._process_req(req)
        self.queue.append(req)

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(self) -> List[Req]:
        """pop the reqs which has finished bootstrapping"""

        self.kv_manager.memdesc_collector.recv_msgs()

        bootstrapped_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            return []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.queue], self.gloo_group
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                raise Exception("Bootstrap failed")
            logger.info(f"[NIXL PD disagg] bqueue poll result: {KVPoll.str(poll)} bootstrap room: {req.bootstrap_room}, ")

            # KV.WaitingForInput - init here
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                logger.info(f"[NIXL PD disagg] pop_bootstrapped: no available metadata buffer index")
                break
            req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
            assert req.metadata_buffer_index is not None
            assert req.disagg_kv_sender
            req.disagg_kv_sender.init(num_kv_indices, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        if len(bootstrapped_reqs) > 0:
            logger.info(f"[NIXL PD disagg] Bbqueue pop out: {len(bootstrapped_reqs)}")
        return bootstrapped_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def process_batch_result_disagg_prefill(
        self: Scheduler, batch: ScheduleBatch, result: GenerationBatchResult # type: ignore
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_infight_queue
        Adapted from process_batch_result_prefill
        """

        next_token_ids = result.next_token_ids.tolist() # type: ignore

        for req, next_token_id in zip(batch.reqs, next_token_ids, strict=True):
            req: Req
            if req.is_chunked <= 0:
                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                self.send_kv_chunk(req, token_id=next_token_id)
                self.disagg_prefill_infight_queue.append(req)
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

        # TODO: Not sure if this is necessary
        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            # We need to remove this for overlap schedule.
            self.current_stream.synchronize()
            assert batch.next_batch_sampling_info.sampling_info_done
            batch.next_batch_sampling_info.sampling_info_done.set()

    def process_disagg_prefill_infight_queue(self: Scheduler) -> None: # type: ignore
        """
        Poll the requests in the middle of transfer. If done, return the request.
        """
        assert len(self.disagg_prefill_infight_queue) > 0

        done_reqs = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_infight_queue],
            self.tp_worker.get_tp_cpu_group(),
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_infight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_infight_queue, polls):
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                raise Exception("Transferring failed")

        # Stream requests which have finished transfer
        self.stream_output(done_reqs, False, None)

        for req in done_reqs:
            self.disagg_prefill_pending_queue.req_to_metadata_buffer_idx_allocator.free(
                req.metadata_buffer_index
            )
            logger.info(f'[NIXL PD disagg] pop_infight_queue broom={req.bootstrap_room}')

        self.disagg_prefill_infight_queue = undone_reqs

    def process_prefill_chunk(self: Scheduler) -> None: # type: ignore
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                self.send_kv_chunk(self.chunked_req)
                # chunked request keeps its rid but will get a new req_pool_idx
                assert self.req_to_token_pool
                assert self.chunked_req.req_pool_idx
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                self.running_batch.batch_is_full = False

    def send_kv_chunk(
        self: Scheduler, req: Req, token_id: Optional[int] = None # type: ignore
    ) -> None:
        """
        Send a prefilled chunk to the decode server

        token_id is the first output token.
        """
        start_idx = req.start_send_idx
        end_idx = min(len(req.fill_ids), len(req.origin_input_ids)) # type: ignore
        assert self.req_to_token_pool is not None
        assert self.req_to_token_pool.req_to_token is not None
        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx][start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        if token_id is not None:
            self.disagg_prefill_pending_queue.allocate_token_id(
                req.metadata_buffer_index, token_id
            )
        logger.info(f"[NIXL PD disagg] send_kv_chunk. Req room: {req.bootstrap_room} kv_indices: {kv_indices}")
        assert req.disagg_kv_sender
        req.disagg_kv_sender.send(kv_indices)
