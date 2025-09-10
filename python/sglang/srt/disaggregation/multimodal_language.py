"""
1. PreallocQueue:

2. TransferQueue:

3. WaitingQueue:

4. RunningBatch:
"""

from __future__ import annotations

import ctypes
import logging
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
    metadata_buffer_index: int = -1


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

    def _init_data_manager(self):
        kv_args = KVArgs()
        kv_args.engine_rank = self.tp_rank
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        data_manager_class = get_kv_class(
            self.transfer_backend, KVClassType.MANAGER, is_multimodal=True
        )
        data_manager = data_manager_class(
            kv_args,
            DisaggregationMode.LANGUAGE,
            self.scheduler.server_args,
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

            language_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc(
                    fake=isinstance(language_req.embedding_receiver, FakeKVReceiver)
                )
            )

            assert language_req.metadata_buffer_index is not None

            language_req.embedding_receiver.init(
                embedding_index=language_req.metadata_buffer_index
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
            language_req.metadata_buffer_index,
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
                idx = language_req.metadata_buffer_index
                if not isinstance(language_req.embedding_receiver, FakeKVReceiver):
                    embedding_data, fill_ids, mrope_positions, aux_datas = (
                        self.metadata_buffers.get_buf(idx)
                    )
                    embedding_length = aux_datas[0]
                    mrope_position_delta = aux_datas[1]
                    language_req.req.input_embeds = embedding_data[:embedding_length, :]
                    mrope_positions = mrope_positions[: 3 * embedding_length].reshape(
                        3, embedding_length
                    )
                    ori_input_length = len(language_req.req.origin_input_ids)
                    language_req.req.origin_input_ids = fill_ids[
                        :embedding_length
                    ].tolist()
                    mm_inputs = None
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
                    # NOTE: we need to set the metadata buffer index to the request
                    # because the metadata buffer index will be freed after the request is done
                    # to avoid embedding buffer is freed before the request is done
                    language_req.req.metadata_buffer_index = (
                        language_req.metadata_buffer_index
                    )
                else:
                    self.req_to_metadata_buffer_idx_allocator.free(idx, fake=True)

                transferred_reqs.append(language_req.req)
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
