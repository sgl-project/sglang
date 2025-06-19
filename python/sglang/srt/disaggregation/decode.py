"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.disaggregation.base import BaseKVManager, BaseKVReceiver, KVPoll
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region():
            self.req_to_token = torch.zeros(
                (size + pre_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(size + pre_alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: BaseKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1


class DecodePreallocQueue:
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        prefill_pp_size: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # this is always a chunk cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.prefill_pp_size = prefill_pp_size
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.prefill_pp_size = prefill_pp_size
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> BaseKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = self.tp_size // self.dp_size
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)
        kv_args.decode_tp_size = attn_tp_size
        kv_args.prefill_pp_size = self.prefill_pp_size
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            self.retracted_queue.append(req)
        else:
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                kv_receiver_class = get_kv_class(
                    TransferBackend.FAKE, KVClassType.RECEIVER
                )
            else:
                kv_receiver_class = get_kv_class(
                    self.transfer_backend, KVClassType.RECEIVER
                )

            kv_receiver = kv_receiver_class(
                mgr=self.kv_manager,
                bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
                bootstrap_room=req.bootstrap_room,
                data_parallel_rank=req.data_parallel_rank,
            )

            self.queue.append(
                DecodeRequest(req=req, kv_receiver=kv_receiver, waiting_for_input=False)
            )

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(self) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens(count_retracted=False)

        for i, req in enumerate(self.retracted_queue):
            if self.req_to_token_pool.available_size() <= 0:
                break

            required_tokens_for_request = (
                len(req.origin_input_ids)
                + len(req.output_ids)
                + self.num_reserved_decode_tokens
            )
            if required_tokens_for_request > allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            allocatable_tokens -= required_tokens_for_request

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(self) -> None:
        if not self.queue:
            return

        if all(decode_req.waiting_for_input for decode_req in self.queue):
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def pop_preallocated(self) -> List[DecodeRequest]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        self._update_handshake_waiters()

        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        allocatable_tokens = self._allocatable_tokens(
            retractable_tokens=retractable_tokens, count_retracted=True
        )
        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                indices_to_remove.add(i)

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = len(decode_req.req.origin_input_ids)
            required_tokens_for_request = (
                origin_input_len + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    + decode_req.req.sampling_params.max_new_tokens
                    - retractable_tokens,
                )
                > allocatable_tokens
            ):
                break
            if required_tokens_for_request > allocatable_tokens:
                break

            allocatable_tokens -= required_tokens_for_request
            self._pre_alloc(decode_req.req)

            kv_indices = (
                self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][
                    : len(decode_req.req.origin_input_ids)
                ]
                .cpu()
                .numpy()
            )

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(
                kv_indices, self.token_to_kv_pool_allocator.page_size
            )
            decode_req.kv_receiver.init(page_indices, decode_req.metadata_buffer_index)
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs

    def _allocatable_tokens(
        self, retractable_tokens: Optional[int] = None, count_retracted: bool = True
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    x.sampling_params.max_new_tokens
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )

        available_size = self.token_to_kv_pool_allocator.available_size()

        allocatable_tokens = available_size - max(
            # preserve some space for future decode
            self.num_reserved_decode_tokens
            * (
                len(self.scheduler.running_batch.reqs)
                + len(self.transfer_queue.queue)
                + len(self.scheduler.waiting_queue)
            ),
            # make sure each request can finish if reach max_tokens with all other requests retracted
            need_space_for_single_req,
        )

        # Note: if the last fake extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_extend()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            allocatable_tokens -= sum(
                [
                    len(req.origin_input_ids)
                    + len(req.output_ids)
                    + self.num_reserved_decode_tokens
                    for req in self.retracted_queue
                ]
            )
        return allocatable_tokens

    def _pre_alloc(self, req: Req) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool"""
        req_pool_indices = self.req_to_token_pool.alloc(1)

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        req.req_pool_idx = req_pool_indices[0]

        if self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(
                len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            )
        else:
            num_tokens = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                prefix_lens=torch.tensor(
                    [0],
                    dtype=torch.int64,
                    device=self.token_to_kv_pool_allocator.device,
                ),
                seq_lens=torch.tensor(
                    [num_tokens],
                    dtype=torch.int64,
                    device=self.token_to_kv_pool_allocator.device,
                ),
                last_loc=torch.tensor(
                    [-1],
                    dtype=torch.int64,
                    device=self.token_to_kv_pool_allocator.device,
                ),
                extend_num_tokens=num_tokens,
            )

        assert (
            kv_loc is not None
        ), "KV cache is full! There is a bug in memory estimation."

        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)

        # populate metadata
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.extend_input_len = len(req.origin_input_ids)

        return kv_loc


class DecodeTransferQueue:
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)

    def pop_transferred(self) -> List[Req]:
        if not self.queue:
            return []
        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Failed:
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                # unlock the kv cache or it will have memory leak
                self.tree_cache.cache_finished_req(decode_req.req)
                indices_to_remove.add(i)
                continue
            elif poll == KVPoll.Success:

                idx = decode_req.metadata_buffer_index
                (
                    output_id,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                ) = self.metadata_buffers.get_buf(idx)

                decode_req.req.output_ids.append(output_id[0].item())

                if decode_req.req.return_logprob:
                    decode_req.req.output_token_logprobs_val.append(
                        output_token_logprobs_val[0].item()
                    )
                    decode_req.req.output_token_logprobs_idx.append(
                        output_token_logprobs_idx[0].item()
                    )
                    decode_req.req.output_top_logprobs_val.append(
                        output_top_logprobs_val[
                            : decode_req.req.top_logprobs_num
                        ].tolist()
                    )
                    decode_req.req.output_top_logprobs_idx.append(
                        output_top_logprobs_idx[
                            : decode_req.req.top_logprobs_num
                        ].tolist()
                    )
                if hasattr(decode_req.kv_receiver, "clear"):
                    decode_req.kv_receiver.clear()
                transferred_reqs.append(decode_req.req)
                indices_to_remove.add(i)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class SchedulerDisaggregationDecodeMixin:

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            prepare_dp_attn_flag = (
                self.server_args.enable_dp_attention
                or self.server_args.enable_sp_layernorm
            )

            if batch:
                # Generate fake extend output.
                if batch.forward_mode.is_extend():
                    # Note: Logprobs should be handled on the prefill engine.
                    self.stream_output(
                        batch.reqs, any(req.return_logprob for req in batch.reqs)
                    )
                    if prepare_dp_attn_flag:
                        self._prepare_idle_batch_and_run(None)
                else:
                    if prepare_dp_attn_flag:
                        self.prepare_dp_attn_batch(batch)
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
            elif prepare_dp_attn_flag:
                batch, _ = self._prepare_idle_batch_and_run(None)

            if batch is None and (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None
        self.last_batch_in_queue = False  # last batch is modified in-place, so we need another variable to track if it's extend

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch
            last_batch_in_queue = False

            prepare_dp_attn_flag = (
                self.server_args.enable_dp_attention
                or self.server_args.enable_sp_layernorm
            )

            if batch:
                # Generate fake extend output.
                if batch.forward_mode.is_extend():
                    # Note: Logprobs should be handled on the prefill engine.
                    self.stream_output(
                        batch.reqs, any(req.return_logprob for req in batch.reqs)
                    )
                    if prepare_dp_attn_flag:
                        batch_, result = self._prepare_idle_batch_and_run(
                            None, delay_process=True
                        )
                        if batch_:
                            result_queue.append((batch_.copy(), result))
                            last_batch_in_queue = True
                else:
                    if prepare_dp_attn_flag:
                        self.prepare_dp_attn_batch(batch)
                    result = self.run_batch(batch)
                    result_queue.append((batch.copy(), result))

                    if (self.last_batch is None) or (not self.last_batch_in_queue):
                        # Create a dummy first batch to start the pipeline for overlap schedule.
                        # It is now used for triggering the sampling_info_done event.
                        tmp_batch = ScheduleBatch(
                            reqs=None,
                            forward_mode=ForwardMode.DUMMY_FIRST,
                            next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                        )
                        self.set_next_batch_sampling_info_done(tmp_batch)
                    last_batch_in_queue = True

            elif prepare_dp_attn_flag:
                batch, result = self._prepare_idle_batch_and_run(
                    None, delay_process=True
                )
                if batch:
                    result_queue.append((batch.copy(), result))
                    last_batch_in_queue = True

            # Process the results of the previous batch but skip if the last batch is extend
            if self.last_batch and self.last_batch_in_queue:
                tmp_batch, tmp_result = result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result(tmp_batch, tmp_result)

            if batch is None and (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            self.last_batch_in_queue = last_batch_in_queue

    def _prepare_idle_batch_and_run(self, batch, delay_process=False):
        batch, _ = self.prepare_dp_attn_batch(batch)
        result = None
        if batch:
            result = self.run_batch(batch)
            if not delay_process:
                self.process_batch_result(batch, result)
        return batch, result

    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[Tuple[ScheduleBatch, bool]]:
        """Create fake completed prefill if possible and merge with running batch"""
        # Merge the prefill batch into the running batch
        last_batch = self.last_batch
        if last_batch and last_batch.forward_mode.is_extend():
            # chunked prefill doesn't happen in decode instance.
            assert self.chunked_req is None
            # Filter finished batches.
            last_batch.filter_batch()
            if not last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = last_batch
                else:
                    # merge running_batch with prefill batch
                    self.running_batch.merge_batch(last_batch)

        new_prebuilt_batch = self.get_new_prebuilt_batch()

        ret: Optional[ScheduleBatch] = None
        if new_prebuilt_batch:
            ret = new_prebuilt_batch
        else:
            if self.running_batch.is_empty():
                ret = None
            else:
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None

        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        if len(self.waiting_queue) == 0:
            return None

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                req.init_next_round_input(self.tree_cache)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt_extend()
        new_batch.process_prebuilt_extend(self.server_args, self.model_config)

        return new_batch

    def process_decode_queue(self: Scheduler):
        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        req_conns = self.disagg_decode_prealloc_queue.pop_preallocated()
        self.disagg_decode_transfer_queue.extend(req_conns)
        alloc_reqs = (
            self.disagg_decode_transfer_queue.pop_transferred()
        )  # the requests which kv has arrived
        self.waiting_queue.extend(alloc_reqs)
