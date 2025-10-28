"""
1. PreallocQueue:

2. TransferQueue:

3. WaitingQueue:

4. RunningBatch:
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Deque, List, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.disaggregation.base import BaseKVReceiver, KVArgs, KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MultimodalDataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    ScheduleBatch,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import DynamicGradMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


@dataclass
class MultimodalLanguageRequest:
    req: Req
    embedding_receiver: BaseKVReceiver
    waiting_for_input: bool = False
    embedding_indices: List[int] = None

    # for resumed transfer
    partial_input_embeds: torch.Tensor = None
    partial_mrope_positions: torch.Tensor = None
    partial_fill_ids: List[int] = None
    partial_sent_tokens: int = None
    partial_aux_datas: torch.Tensor = None
    partial_deepstack_embedding: torch.Tensor = None


class MultimodalLanguagePreallocQueue:
    def __init__(
        self,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MultimodalDataBuffers,
        tp_rank: int,
        tp_size: int,
        scheduler: Scheduler,
        transfer_backend: TransferBackend,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
    ):
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.transfer_backend = transfer_backend

        self.data_manager = self._init_data_manager()
        self.bootstrap_port = bootstrap_port
        self.queue: List[MultimodalLanguageRequest] = []
        self.gloo_group = gloo_group

        # Get default buffer size from environment variable
        # Language side only sees text, not the full embedding length from encode side
        # So we use a default buffer size (can be configured via env var)
        self.default_allocate_tokens = int(
            os.getenv(
                "SGLANG_EMBEDDING_DEFAULT_ALLOCATE_BUFFER_SIZE",
                "8192",
            )
        )

    def _init_data_manager(self):
        kv_args = KVArgs()
        kv_args.engine_rank = self.tp_rank
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        # Set required fields for multimodal (language mode doesn't use KV cache)
        kv_args.kv_data_ptrs = []
        kv_args.kv_data_lens = []
        kv_args.kv_item_lens = []
        kv_args.decode_tp_size = 0
        kv_args.kv_head_num = 0
        kv_args.page_size = 0
        kv_args.prefill_pp_size = 1
        kv_args.pp_rank = 0
        kv_args.prefill_start_layer = 0
        kv_args.system_dp_rank = 0

        data_manager_class = get_kv_class(
            self.transfer_backend, KVClassType.MANAGER, is_multimodal=True
        )
        data_manager = data_manager_class(
            kv_args,
            DisaggregationMode.LANGUAGE,
            self.scheduler.server_args,
            is_multimodal=True,
        )
        return data_manager

    def add(self, req: Req):
        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            # Fake transfer for warmup reqs
            embedding_receiver_class = get_kv_class(
                TransferBackend.FAKE, KVClassType.RECEIVER, is_multimodal=True
            )
        else:
            embedding_receiver_class = get_kv_class(
                self.transfer_backend, KVClassType.RECEIVER, is_multimodal=True
            )
        embedding_receiver = embedding_receiver_class(
            mgr=self.data_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            prefill_dp_rank=req.data_parallel_rank,
        )
        self.queue.append(
            MultimodalLanguageRequest(req=req, embedding_receiver=embedding_receiver)
        )

    def extend(self, reqs: List[Req]):
        for req in reqs:
            self.add(req)

    def _update_handshake_waiters(self) -> None:
        if not self.queue:
            return

        if all(decode_req.waiting_for_input for decode_req in self.queue):
            return

        polls = poll_and_all_reduce(
            [language_req.embedding_receiver for language_req in self.queue],
            self.gloo_group,
        )

        for i, (language_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                language_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                error_message = f"MultimodalLanguage handshake failed for request rank={self.tp_rank} {language_req.req.rid=} {language_req.req.bootstrap_room=}"
                try:
                    language_req.embedding_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    language_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    def pop_preallocated(self):
        self._update_handshake_waiters()

        preallocated_reqs = []
        indices_to_remove = set()

        for i, language_req in enumerate(self.queue):
            if isinstance(language_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [language_req.req], language_req.req.return_logprob
                )
                indices_to_remove.add(i)

        for i, language_req in enumerate(self.queue):
            if i in indices_to_remove:
                continue

            if not language_req.waiting_for_input:
                continue

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            # Language side: allocate blocks based on default buffer size
            # Since we only have text here, not the full embedding from encode side
            language_req.embedding_indices = (
                self.req_to_metadata_buffer_idx_allocator.alloc(
                    num_tokens=self.default_allocate_tokens,
                    req_id=language_req.req.rid,
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
                )
            )

            if language_req.embedding_indices is None:
                break

            # Calculate actual allocated tokens from allocated blocks
            actual_allocated_tokens = (
                len(language_req.embedding_indices) * self.metadata_buffers.block_size
            )
            # Initialize receiver with block_indices and allocated_tokens
            language_req.embedding_receiver.init(
                embedding_indices=language_req.embedding_indices,
                allocated_tokens=actual_allocated_tokens,
            )
            preallocated_reqs.append(language_req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs


class MultimodalLanguageTransferQueue:
    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MultimodalDataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[MultimodalLanguageRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache

    def add(self, req):
        self.queue.append(req)

    def extend(self, reqs):
        self.queue.extend(reqs)

    def _handle_failed_request(self, language_req: MultimodalLanguageRequest):
        error_message = f"MultiModalLanguage transfer failed for request rank={self.scheduler.tp_rank} {language_req.req.rid=} {language_req.req.bootstrap_room=}"
        try:
            language_req.embedding_receiver.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
        logger.error(error_message)
        prepare_abort(
            language_req.req,
            error_message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        self.scheduler.stream_output(
            [language_req.req], language_req.req.return_logprob
        )
        # unlock the kv cache or it will have memory leak
        self.req_to_metadata_buffer_idx_allocator.free(
            block_indices=language_req.embedding_indices,
            req_id=language_req.req.rid,
            fake=isinstance(language_req.embedding_receiver, FakeKVReceiver),
        )
        if self.scheduler.enable_metrics:
            self.scheduler.metrics_collector.increment_transfer_failed_reqs()

    def pop_transferred(self):
        if not self.queue:
            return []

        polls = poll_and_all_reduce(
            [language_req.embedding_receiver for language_req in self.queue],
            self.gloo_group,
        )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (language_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Failed:
                self._handle_failed_request(language_req)
                # unlock the kv cache or it will have memory leak
                indices_to_remove.add(i)
                continue
            elif poll == KVPoll.Success:
                # Use block_indices instead of single index
                block_indices = language_req.embedding_indices
                if not isinstance(language_req.embedding_receiver, FakeKVReceiver):
                    # Check if this is a resumed transfer (has partial data)
                    if language_req.partial_input_embeds is not None:
                        # For resumed transfer, get the remaining data based on actual total length
                        actual_total_length = (
                            language_req.partial_aux_datas[0].item()
                            - language_req.partial_sent_tokens
                        )
                        (
                            embedding_data,
                            fill_ids,
                            mrope_positions,
                            aux_datas,
                            deepstack_embedding,
                        ) = self.metadata_buffers.get_buf(
                            block_indices=block_indices,
                            actual_total_length=actual_total_length,
                        )
                        # Merge partial data with new data
                        logger.debug(
                            f"Merging resumed transfer data for rid={language_req.req.rid}"
                        )

                        # Concatenate embeddings
                        embedding_data = torch.cat(
                            [language_req.partial_input_embeds, embedding_data]
                        )
                        if deepstack_embedding is not None:
                            deepstack_embedding = torch.cat(
                                [
                                    language_req.partial_deepstack_embedding,
                                    deepstack_embedding,
                                ]
                            )

                        # Concatenate fill_ids
                        fill_ids = torch.cat(
                            [torch.tensor(language_req.partial_fill_ids), fill_ids]
                        )

                        # Concatenate mrope_positions
                        mrope_positions = torch.cat(
                            [language_req.partial_mrope_positions, mrope_positions],
                            dim=-1,
                        )

                        aux_datas = language_req.partial_aux_datas.clone()

                        # Clean up partial data
                        del language_req.partial_input_embeds
                        del language_req.partial_fill_ids
                        del language_req.partial_mrope_positions
                        del language_req.partial_sent_tokens
                        del language_req.partial_deepstack_embedding
                    else:
                        (
                            embedding_data,
                            fill_ids,
                            mrope_positions,
                            aux_datas,
                            deepstack_embedding,
                        ) = self.metadata_buffers.get_buf(block_indices=block_indices)

                    embedding_length = int(aux_datas[0])
                    mrope_position_delta = aux_datas[1]
                    mm_inputs = None
                    ori_input_length = len(language_req.req.origin_input_ids)
                    language_req.req.origin_input_ids = fill_ids.tolist()

                    if deepstack_embedding is not None:
                        # NOTE: merge input_embeds and deepstack_embedding to input_embeds to
                        # simplify the model forward pass
                        language_req.req.input_embeds = torch.cat(
                            [embedding_data, deepstack_embedding],
                            dim=-1,
                        ).contiguous()
                    else:
                        language_req.req.input_embeds = embedding_data

                    if ori_input_length == embedding_length:
                        mm_inputs = None
                    elif ori_input_length < embedding_length:
                        # NOTE: mock mm_inputs to make mm_inputs not None
                        # need to be checked carefully for modality-attributes
                        mm_inputs = MultimodalInputs(
                            mm_items=[
                                MultimodalDataItem(
                                    modality=Modality.IMAGE, model_specific_data={}
                                ),
                            ]
                        )
                        mm_inputs.mrope_positions = mrope_positions
                        mm_inputs.mrope_position_delta = torch.tensor(
                            [mrope_position_delta]
                        ).unsqueeze(1)
                    else:
                        # take as transfer failed case
                        self._handle_failed_request(language_req)
                        indices_to_remove.add(i)
                        continue
                    language_req.req.multimodal_inputs = mm_inputs
                    # NOTE: we need to set the metadata block indices to the request
                    # because the metadata buffer should be freed after the request prefill forward finished
                    language_req.req.embedding_indices = language_req.embedding_indices
                else:
                    self.req_to_metadata_buffer_idx_allocator.free(
                        block_indices=block_indices, fake=True
                    )

                transferred_reqs.append(language_req.req)
                indices_to_remove.add(i)
            elif poll == KVPoll.Transferring:
                # Partial transfer complete, need to resume with remaining data
                block_indices = language_req.embedding_indices
                if (
                    not isinstance(language_req.embedding_receiver, FakeKVReceiver)
                    and language_req.partial_input_embeds is None
                ):
                    # Get partial data and actual total length from aux_datas
                    (
                        embedding_data,
                        fill_ids,
                        mrope_positions,
                        aux_datas,
                        deepstack_embedding,
                    ) = self.metadata_buffers.get_buf(block_indices=block_indices)
                    actual_total_length = int(aux_datas[0])  # Actual total length
                    sent_tokens = len(fill_ids)  # Tokens already sent

                    if actual_total_length > sent_tokens:
                        # Need to resume transfer
                        remaining_tokens = actual_total_length - sent_tokens

                        logger.debug(
                            f"Partial transfer detected for rid={language_req.req.rid}: "
                            f"received {sent_tokens}/{actual_total_length} tokens, "
                            f"need to resume for {remaining_tokens} more tokens"
                        )

                        # Allocate new space for remaining tokens first
                        new_allocation = (
                            self.req_to_metadata_buffer_idx_allocator.alloc(
                                num_tokens=remaining_tokens,
                                req_id=language_req.req.rid,
                                fake=isinstance(
                                    language_req.embedding_receiver, FakeKVReceiver
                                ),
                            )
                        )

                        if new_allocation is None:
                            # Not enough memory to resume now, wait for next iteration
                            logger.debug(
                                f"Waiting for memory to resume transfer for rid={language_req.req.rid}, "
                                f"need {remaining_tokens} tokens"
                            )
                            # Keep request in queue, will retry in next pop_transferred call
                            # Don't cache partial data or free old allocation yet
                            continue

                        # Only after successful allocation, cache partial data and free old allocation
                        language_req.partial_input_embeds = embedding_data
                        language_req.partial_fill_ids = fill_ids.tolist()
                        language_req.partial_mrope_positions = mrope_positions
                        language_req.partial_aux_datas = aux_datas
                        language_req.partial_sent_tokens = sent_tokens
                        language_req.partial_deepstack_embedding = deepstack_embedding

                        # Free old allocation
                        self.req_to_metadata_buffer_idx_allocator.free(
                            block_indices=block_indices,
                            req_id=language_req.req.rid,
                            fake=isinstance(
                                language_req.embedding_receiver, FakeKVReceiver
                            ),
                        )

                        # Update embedding_indices
                        language_req.embedding_indices = new_allocation

                        # Calculate allocated_tokens from new allocation
                        block_size = self.metadata_buffers.block_size
                        allocated_tokens = len(new_allocation) * block_size

                        # Send resume request
                        language_req.embedding_receiver.resume_transfer(
                            embedding_indices=new_allocation,
                            sent_tokens=sent_tokens,
                            allocated_tokens=allocated_tokens,
                        )

                        logger.debug(
                            f"Resume transfer initiated for rid={language_req.req.rid}: "
                            f"allocated {len(new_allocation)} blocks ({allocated_tokens} tokens)"
                        )
                    else:
                        # This shouldn't happen - Transferring status but all data received
                        logger.warning(
                            f"Unexpected: Transferring status but sent_tokens={sent_tokens} >= "
                            f"actual_total_length={actual_total_length}"
                        )
                # Continue waiting for transfer to complete
                pass
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            block_indices = self.queue[i].embedding_indices
            assert block_indices is not None and len(block_indices) > 0

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        return transferred_reqs


class SchedulerDisaggregationMultiModalLanguageMixin:

    @DynamicGradMode()
    def event_loop_normal_disagg_multimodal_language(self: Scheduler):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_multimodal_language_queue()
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    @DynamicGradMode()
    def event_loop_overlap_disagg_multimodal_language(self: Scheduler):
        self.result_queue: Deque[Tuple[ScheduleBatch, GenerationBatchResult]] = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_multimodal_language_queue()
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            batch_result = None
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.launch_batch_sample_if_needed(batch_result)
            self.last_batch = batch

    def process_multimodal_language_queue(self: Scheduler):
        req_conns = self.disagg_language_prealloc_queue.pop_preallocated()
        self.disagg_language_transfer_queue.extend(req_conns)
        alloc_reqs = self.disagg_language_transfer_queue.pop_transferred()
        self.waiting_queue.extend(alloc_reqs)
