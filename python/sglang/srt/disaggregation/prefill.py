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
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.disaggregation.base import BaseKVManager, KVArgs, KVPoll
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FakeBootstrapHost,
    KVClassType,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    kv_to_page_indices,
    kv_to_page_num,
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
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        transfer_backend: TransferBackend,
        scheduler: Scheduler,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.aux_dtype = aux_dtype

        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.transfer_backend = transfer_backend
        self.scheduler = scheduler
        self.kv_manager = self._init_kv_manager()
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.bootstrap_port = bootstrap_port

    def store_prefill_results(self, idx: int, token_id: int):
        assert token_id >= 0, f"token_id: {token_id} is negative"
        output_id_buffer = self.metadata_buffers[0]
        output_id_buffer[idx] = token_id

    def _init_kv_manager(self) -> BaseKVManager:
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
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args, DisaggregationMode.PREFILL, self.scheduler.server_args
        )
        return kv_manager

    def add(self, req: Req) -> None:
        if req.bootstrap_host == FakeBootstrapHost:
            # Fake transfer for warmup reqs
            kv_sender_class = get_kv_class(TransferBackend.FAKE, KVClassType.SENDER)
        else:
            kv_sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)
        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
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

            # KV.WaitingForInput
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None
            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return bootstrapped_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler):
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            # Handle DP attention
            if (
                self.server_args.enable_dp_attention
                or self.server_args.enable_sp_layernorm
            ):
                batch, _ = self.prepare_dp_attn_batch(batch)

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result_disagg_prefill(batch, result)

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler):
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            # Handle DP attention
            if (
                self.server_args.enable_dp_attention
                or self.server_args.enable_sp_layernorm
            ):
                batch, _ = self.prepare_dp_attn_batch(batch)

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result_disagg_prefill(tmp_batch, tmp_result)

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """

        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            bid,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.bid,
        )

        # Transfer kv for prefill completed requests and add it into disagg_prefill_infight_queue
        if self.enable_overlap:
            # wait
            _, next_token_ids = self.tp_worker.resolve_last_batch_result(launch_done)
        else:
            next_token_ids = result.next_token_ids.tolist()

        for req, next_token_id in zip(batch.reqs, next_token_ids, strict=True):
            req: Req
            if req.is_chunked <= 0:
                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                self.send_kv_chunk(req, token_id=next_token_id)
                self.disagg_prefill_inflight_queue.append(req)
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

                if self.enable_overlap:
                    self.send_kv_chunk(req, end_idx=req.tmp_end_idx)

    def process_disagg_prefill_inflight_queue(self: Scheduler) -> None:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        """
        assert len(self.disagg_prefill_inflight_queue) > 0

        done_reqs = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                raise Exception("Transferring failed")

        for req in done_reqs:
            self.disagg_prefill_bootstrap_queue.req_to_metadata_buffer_idx_allocator.free(
                req.metadata_buffer_index
            )

        # Stream requests which have finished transfer
        self.stream_output(done_reqs, False, None)

        self.disagg_prefill_inflight_queue = undone_reqs

    def process_prefill_chunk(self: Scheduler) -> None:
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                if self.enable_overlap:
                    # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                    self.chunked_req.tmp_end_idx = min(
                        len(self.chunked_req.fill_ids),
                        len(self.chunked_req.origin_input_ids),
                    )
                else:
                    self.send_kv_chunk(self.chunked_req)
                # chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                self.running_batch.batch_is_full = False

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        token_id: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        # if end_idx is specified, use it as the end index of the kv chunk because in overlap schedule,
        # the resolved length is not the same as fill_ids's length
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )
        last_chunk = token_id is not None

        if (not last_chunk) and (
            end_idx % page_size != 0
        ):  # todo: remove the second condition
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        # Update next start_send_idx
        req.start_send_idx = end_idx

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        if last_chunk is True:
            self.disagg_prefill_bootstrap_queue.store_prefill_results(
                req.metadata_buffer_index, token_id
            )
        page_indices = kv_to_page_indices(kv_indices, page_size)

        page_start_idx = start_idx // page_size
        page_end_idx = page_start_idx + len(page_indices)

        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty"
            )
            return

        req.disagg_kv_sender.send(
            page_indices, slice(page_start_idx, page_end_idx), last_chunk
        )
