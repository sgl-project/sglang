"""
1. Bootstrap Queue

2. Waiting Queue

3. Inflight Queue

"""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import torch
from torch.distributed import ProcessGroup

from sglang.srt.disaggregation.base import BaseKVManager, KVArgs, KVPoll
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FAKE_BOOTSTRAP_HOST,
    KVClassType,
    MetaMultiModaldataBuffers,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)


class MultimodalEmbeddingBootstrapQueue:
    def __init__(
        self,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetaMultiModaldataBuffers,
        tp_rank: int,
        tp_size: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        transfer_backend: TransferBackend,
        scheduler: Scheduler,
    ):
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.transfer_backend = transfer_backend
        self.scheduler = scheduler
        self.data_manager = self._init_data_manager()
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.bootstrap_port = bootstrap_port

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
            DisaggregationMode.EMBEDDING,
            self.scheduler.server_args,
        )
        logger.debug(f"type(data_manager): {type(data_manager)}")
        logger.debug(
            f"init MultimodalEmbedding data manager: {self.tp_rank=}; {self.tp_size=}; {self.transfer_backend=}; {self.scheduler.server_args.disaggregation_ib_device=}"
        )
        return data_manager

    def add(self, req: Req):
        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            # Fake transfer for warmup reqs
            embedding_sender_class = get_kv_class(
                TransferBackend.FAKE, KVClassType.SENDER, is_multimodal=True
            )
        else:
            embedding_sender_class = get_kv_class(
                self.transfer_backend, KVClassType.SENDER, is_multimodal=True
            )
        logger.debug(
            f"Add request to MultimodalEmbedding bootstrap queue: {self.data_manager.request_status=}"
        )
        req.disagg_embedding_sender = embedding_sender_class(
            mgr=self.data_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
        )
        self.queue.append(req)

    def extend(self, reqs: List[Req]):
        for req in reqs:
            self.add(req)

    def pop_bootstrapped(self):
        """pop the reqs which has finished bootstrapping"""
        bootstrapped_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            return []

        polls = poll_and_all_reduce(
            [req.disagg_embedding_sender for req in self.queue], self.gloo_group
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            # logger.debug(f"MultimodalEmbedding bootstrap poll for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=} {poll=}")
            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"MultimodalEmbedding bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_embedding_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                # TODO: check embedding return_logprob
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                continue

            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None
            logger.debug(
                f"MultimodalEmbedding bootstrap success for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
            )
            req.disagg_embedding_sender.init(embedding_index=req.metadata_buffer_index)
            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        if bootstrapped_reqs:
            logger.debug(
                f"MultimodalEmbedding bootstrap queue size: {len(bootstrapped_reqs)}"
            )
        return bootstrapped_reqs


class SchedulerDisaggregationMultimodalEmbeddingMixin:

    @torch.no_grad()
    def event_loop_normal_disagg_multimodal_embedding(self: Scheduler):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_embedding_bootstrap_queue.pop_bootstrapped()
            )
            # batch = self.get_next_batch_to_run()
            batch = self.get_new_batch_prefill()
            # batch = self.get_new_batch_multimodal_embedding()

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                logger.debug(
                    f"End to run batch {[req.bootstrap_room for req in batch.reqs]}"
                )
                self.process_batch_result_disagg_multimodal_embedding(batch, result)

            if len(self.disagg_embedding_inflight_queue) > 0:
                self.process_multimodal_embedding_inflight_queue()

            if batch is None and len(self.disagg_embedding_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
            self.running_batch.batch_is_full = False

    def process_batch_result_disagg_multimodal_embedding(
        self: Scheduler, batch: ScheduleBatch, result: EmbeddingBatchResult
    ):
        """
        Transfer embedding results for completed requests and add it into disagg_multimodal_embedding_inflight_queue
        """
        # TODO: support overlap scheduler
        # TODO: check chunk forward for vision embedding
        # TODO: support batch embedding

        embeddings = result.embeddings
        # if len(batch.reqs) > 1:
        #     raise NotImplementedError(
        #         "Multimodal embedding with multiple requests is not supported"
        #     )

        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.embedding = embeddings[i]
            if req.is_chunked <= 0:
                # Dummy output token for embedding models
                req.output_ids.append(0)
                self.tree_cache.cache_unfinished_req(req)
                self.disagg_embedding_inflight_queue.append(req)
                self.send_embedding_chunk(req, last_chunk=True)

                if req.grammar is not None:
                    req.grammar.accept_token(0)
                    req.grammar.finished = req.finished()
            else:
                raise NotImplementedError(
                    "Chunked forward for multimodal embedding is not supported"
                )

        # self.stream_output(batch.reqs, batch.return_logprob, skip_req=None)

    def process_multimodal_embedding_inflight_queue(self: Scheduler):
        """
        Poll the requests in the middle of transfer. If done, return the request.
        """
        assert len(self.disagg_embedding_inflight_queue) > 0

        done_reqs = []

        # NOTE: no need poll_and_all_reduce for vision embedding, because all data is same
        polls = poll_and_all_reduce(
            [
                req.disagg_embedding_sender
                for req in self.disagg_embedding_inflight_queue
            ],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_embedding_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_embedding_inflight_queue, polls):
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_embedding_sender, "clear"):
                    req.disagg_embedding_sender.clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                error_message = f"MultimodalEmbedding transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_embedding_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)

        for req in done_reqs:
            self.disagg_embedding_bootstrap_queue.req_to_metadata_buffer_idx_allocator.free(
                req.metadata_buffer_index
            )
            # clean embedding to reduce latency for embedding
            req.embedding = [0]

        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        self.disagg_embedding_inflight_queue = undone_reqs

    def get_new_batch_multimodal_embedding(self: Scheduler):
        # TODO: check necessary
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if self.get_num_allocatable_reqs(running_bs) <= 0 and not self.chunked_req:
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            # check for completion of hierarchical cache activities to release memory
            self.tree_cache.writing_check()
            self.tree_cache.loading_check()

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.lora_paths:
            lora_set = set([req.lora_path for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if (
                self.lora_paths
                and len(
                    lora_set
                    | set([req.lora_path for req in adder.can_run_list])
                    | set([req.lora_path])
                )
                > self.max_loras_per_batch
            ):
                self.running_batch.batch_is_full = True
                break

            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
                break

            if self.disaggregation_mode in [
                DisaggregationMode.PREFILL,
                DisaggregationMode.EMBEDDING,
            ]:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True
                    break

            req.init_next_round_input(
                None if prefix_computed else self.tree_cache,
                self.enable_hierarchical_cache,
            )

            res = adder.add_one_req(
                req, self.chunked_req, self.enable_hierarchical_cache
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.queue_time_end = time.perf_counter()

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if self.enable_hierarchical_cache:
            self.tree_cache.ready_to_load_cache()

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.attn_tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            chunked_req=self.chunked_req,
        )
        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def send_embedding_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
    ):
        assert last_chunk == True
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)
            chunk_info = self.disagg_metadata_buffers.get_buf_chunk_info(req)
        req.disagg_embedding_sender.send_embedding(req.metadata_buffer_index, last_chunk, chunk_info)

    def get_num_allocatable_reqs(self: Scheduler, running_bs: int):
        return global_server_args_dict["max_micro_batch_size"] - running_bs
