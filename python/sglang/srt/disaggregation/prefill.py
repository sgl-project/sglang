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

import asyncio
import base64
import logging
import pickle
import threading
from collections import defaultdict, deque
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.disaggregation.base import BaseKVManager, BaseKVSender, KVArgs, KVPoll
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    RemotePrefillReq,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import require_mlp_sync

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

try:
    import nats

    IS_REMOTE_PREFILL_SUPPORT = True
except ImportError:
    logger.info("etcd3 or nats is not installed, remote prefill will not be available.")
    IS_REMOTE_PREFILL_SUPPORT = False

REQ_KV_SENDR_MAP: Dict[str, BaseKVSender] = defaultdict()


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        decode_tp_size: int,
        decode_dp_size: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
        is_remote_prefill: bool = False,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.decode_tp_size = decode_tp_size
        self.decode_dp_size = decode_dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()
        self.is_remote_prefill = is_remote_prefill

    def store_prefill_results(self, idx: int, token_id: int):
        assert token_id >= 0, f"token_id: {token_id} is negative"
        output_id_buffer = self.metadata_buffers[0]
        output_id_buffer[idx] = token_id

    def _init_kv_manager(self) -> BaseKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.decode_tp_size = self.decode_tp_size // self.decode_dp_size
        kv_args.prefill_pp_size = self.pp_size
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
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        return kv_manager

    def add(self, req: Req, num_kv_heads: int) -> None:
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            kv_sender_class = get_kv_class(TransferBackend.FAKE, KVClassType.SENDER)
        else:
            kv_sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        REQ_KV_SENDR_MAP[req.rid] = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        polls = poll_and_all_reduce(
            [REQ_KV_SENDR_MAP[req.rid] for req in self.queue], self.gloo_group
        )
        for i, (req, poll) in enumerate(zip(self.queue, polls)):

            if rids_to_check is not None:
                # if req not in reqs_info_to_check, skip
                if req.rid not in rids_to_check:
                    continue
                # Either waiting for input or failed
                assert poll == KVPoll.WaitingForInput or poll == KVPoll.Failed

            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    REQ_KV_SENDR_MAP[req.rid].failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                continue

            # KV.WaitingForInput - init here
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None

            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)
            REQ_KV_SENDR_MAP[req.rid].init(num_pages, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        # only start the queue thread on rank 0
        if (
            self.is_remote_prefill
            and IS_REMOTE_PREFILL_SUPPORT
            and self.attn_tp_rank == 0
        ):
            self._start_queue_thread()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result_disagg_prefill(batch, result)

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()
        if self.is_remote_prefill and self.attn_tp_rank == 0:
            self._start_queue_thread()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.set_next_batch_sampling_info_done(tmp_batch)

            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result_disagg_prefill(tmp_batch, tmp_result)

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

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
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        if self.enable_overlap:
            # wait
            logits_output, next_token_ids, _ = self.tp_worker.resolve_last_batch_result(
                launch_done
            )
        else:
            next_token_ids = result.next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

        hidden_state_offset = 0
        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            req: Req
            if req.is_chunked <= 0:
                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                self.disagg_prefill_inflight_queue.append(req)
                if logits_output.hidden_states is not None:
                    last_hidden_index = (
                        hidden_state_offset + extend_input_len_per_req[i] - 1
                    )
                    req.hidden_states_tensor = (
                        logits_output.hidden_states[last_hidden_index].cpu().clone()
                    )
                    hidden_state_offset += extend_input_len_per_req[i]
                else:
                    req.hidden_states_tensor = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                self.send_kv_chunk(req, last_chunk=True)

                if req.grammar is not None:
                    req.grammar.accept_token(next_token_id)
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                if self.enable_overlap:
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)

        # We need to remove the sync in the following function for overlap schedule.
        self.set_next_batch_sampling_info_done(batch)

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        polls = poll_and_all_reduce(
            [REQ_KV_SENDR_MAP[req.rid] for req in self.disagg_prefill_inflight_queue],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                assert poll == KVPoll.Success or poll == KVPoll.Failed

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(REQ_KV_SENDR_MAP[req.rid], "clear"):
                    REQ_KV_SENDR_MAP[req.rid].clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    REQ_KV_SENDR_MAP[req.rid].failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
            else:
                assert False, f"Unexpected polling state {poll=}"

        for req in done_reqs:
            del REQ_KV_SENDR_MAP[req.rid]
            self.disagg_prefill_bootstrap_queue.req_to_metadata_buffer_idx_allocator.free(
                req.metadata_buffer_index
            )

        # Don't send the remote prefill request to detokenizer
        if self.is_remote_prefill:
            done_reqs = [req for req in done_reqs if not req.is_remote_prefill]

        # Stream requests which have finished transfer
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req
            self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        """
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.tp_worker.get_tp_group().cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

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
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)
        page_indices = kv_to_page_indices(kv_indices, page_size)
        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty"
            )
            return
        REQ_KV_SENDR_MAP[req.rid].send(page_indices)

    def handle_remote_prefill_req(
        self: Scheduler, remote_prefill_req: RemotePrefillReq
    ):
        import json

        import numpy as np

        from .nixl.conn import KVArgsRegisterInfo, NixlKVSender, TransferInfo

        req = Req(
            rid=remote_prefill_req.rid,
            origin_input_text=remote_prefill_req.origin_input_text,
            origin_input_ids=remote_prefill_req.origin_input_ids,
            sampling_params=remote_prefill_req.sampling_params,
            bootstrap_room=remote_prefill_req.bootstrap_room,
            bootstrap_host=remote_prefill_req.rank_ip,
            bootstrap_port=remote_prefill_req.rank_port,
            is_remote_prefill=True,
        )

        bootstrap_room = req.bootstrap_room
        engine_id = remote_prefill_req.engine_id
        self.disagg_prefill_bootstrap_queue.add(
            req, self.model_config.num_key_value_heads
        )
        kv_sender = REQ_KV_SENDR_MAP[req.rid]
        kv_indices = base64.b64decode(remote_prefill_req.kv_indices)

        if self.remote_engine_configs.get(engine_id) is None:
            # only load on first time
            data = self.etcd_client.get(f"/decode/{self.model_name_hash}/{engine_id}")
            engine_config = json.loads(data[0])
            self.remote_engine_configs[engine_id] = engine_config

        engine_config = self.remote_engine_configs.get(engine_id)
        prefill_per_dp_tp_size = self.tp_size // self.dp_size
        decode_per_dp_tp_size = engine_config["tp_size"] // engine_config["dp_size"]

        # different tp rank per dp group is not allowed in non-MLA backend
        if not self.disagg_prefill_bootstrap_queue.is_mla_backend:
            assert (
                prefill_per_dp_tp_size == decode_per_dp_tp_size
            ), f"Prefill tp size {prefill_per_dp_tp_size} is not equal to decode tp size {decode_per_dp_tp_size} per dp group for non-MLA backend"
            # calculate the target tp rank for the decode server
            target_tp_rank = (
                self.tp_rank % prefill_per_dp_tp_size + remote_prefill_req.engine_rank
            )
            remote_agent_key = f"{engine_id}_{target_tp_rank}"
            if self.remote_agent_map.get(remote_agent_key) is None:
                # only load on first time
                data = self.etcd_client.get(
                    f"/decode/{self.model_name_hash}/{engine_id}/{target_tp_rank}"
                )
                agent_info = json.loads(data[0])
                agent_info["agent_metadata"] = base64.b64decode(
                    agent_info["agent_metadata"]
                )
                self.remote_agent_map[remote_agent_key] = agent_info

            agent_info = self.remote_agent_map.get(remote_agent_key)
            assert agent_info is not None, "Invalid remote agent"
            assert isinstance(
                kv_sender, NixlKVSender
            ), f"Expect NixlKVSender but got {type(kv_sender)} for remote prefill request {remote_prefill_req.rid}"

            agent_name = agent_info["agent_name"]

            logger.debug(f"rank {self.tp_rank} with agent name: {agent_name}")
            kv_mgr = kv_sender.kv_mgr
            if bootstrap_room not in kv_mgr.decode_kv_args_table:
                kv_mgr.decode_kv_args_table[bootstrap_room] = {}
            kv_mgr._add_remote_peer(
                decode_kv_args=KVArgsRegisterInfo(
                    room=remote_prefill_req.bootstrap_room,
                    endpoint=remote_prefill_req.rank_ip,
                    dst_port=remote_prefill_req.rank_port,
                    agent_metadata=agent_info["agent_metadata"],
                    dst_kv_ptrs=agent_info["kv_data_ptrs"],
                    dst_aux_ptrs=agent_info["aux_data_ptrs"],
                    gpu_id=agent_info["gpu_id"],
                )
            )

            if bootstrap_room not in kv_mgr.transfer_infos:
                kv_mgr.transfer_infos[bootstrap_room] = {}
            kv_mgr.transfer_infos[bootstrap_room][agent_key] = TransferInfo(
                room=remote_prefill_req.bootstrap_room,
                endpoint=remote_prefill_req.rank_ip,
                dst_port=remote_prefill_req.rank_port,
                agent_name=agent_name,
                dst_kv_indices=dst_kv_indices,
                dst_aux_index=remote_prefill_req.aux_index,
                required_dst_info_num=0,
            )
            kv_mgr.update_status(bootstrap_room, KVPoll.WaitingForInput)
        else:
            # first tp rank in each dp group will send kv chunk to all decode ranks
            target_tp_ranks = [
                offset + remote_prefill_req.engine_rank
                for offset in range(decode_per_dp_tp_size)
            ]
            for target_tp_rank in target_tp_ranks:
                remote_agent_key = f"{engine_id}_{target_tp_rank}"
                if self.remote_agent_map.get(remote_agent_key) is None:
                    # only load on first time
                    data = self.etcd_client.get(
                        f"/decode/{self.model_name_hash}/{engine_id}/{target_tp_rank}"
                    )
                    agent_info = json.loads(data[0])
                    agent_info["agent_metadata"] = base64.b64decode(
                        agent_info["agent_metadata"]
                    )
                    self.remote_agent_map[remote_agent_key] = agent_info

                agent_info = self.remote_agent_map.get(remote_agent_key)
                assert agent_info is not None, "Invalid remote agent"
                assert isinstance(
                    kv_sender, NixlKVSender
                ), f"Expect NixlKVSender but got {type(kv_sender)} for remote prefill request {remote_prefill_req.rid}"

                dst_kv_indices = (
                    np.frombuffer(kv_indices, dtype=np.int32)
                    if self.attn_tp_rank == 0
                    else np.array([], dtype=np.int32)
                )

                agent_name = agent_info["agent_name"]
                agent_key = f"{agent_name}_{target_tp_rank}"
                logger.debug(f"rank {self.tp_rank} with agent name: {agent_name}")
                kv_mgr = kv_sender.kv_mgr

                if bootstrap_room not in kv_mgr.decode_kv_args_table:
                    kv_mgr.decode_kv_args_table[bootstrap_room] = {}
                kv_mgr._add_remote_peer(
                    decode_kv_args=KVArgsRegisterInfo(
                        room=remote_prefill_req.bootstrap_room,
                        endpoint=remote_prefill_req.rank_ip,
                        dst_port=remote_prefill_req.rank_port,
                        agent_name=agent_name,
                        agent_metadata=agent_info["agent_metadata"],
                        dst_kv_ptrs=agent_info["kv_data_ptrs"],
                        dst_aux_ptrs=agent_info["aux_data_ptrs"],
                        gpu_id=agent_info["gpu_id"],
                    )
                )

                if bootstrap_room not in kv_mgr.transfer_infos:
                    kv_mgr.transfer_infos[bootstrap_room] = {}
                kv_mgr.transfer_infos[bootstrap_room][agent_key] = TransferInfo(
                    room=remote_prefill_req.bootstrap_room,
                    endpoint=remote_prefill_req.rank_ip,
                    dst_port=remote_prefill_req.rank_port,
                    agent_name=agent_name,
                    dst_kv_indices=dst_kv_indices,
                    dst_aux_index=remote_prefill_req.aux_index,
                    required_dst_info_num=0,
                )

                kv_mgr.update_status(bootstrap_room, KVPoll.WaitingForInput)

    def _start_queue_thread(self: Scheduler):
        """Start a thread to send requests to the queue."""

        def start_async_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        event_loop = asyncio.new_event_loop()
        threading.Thread(target=start_async_loop, args=(event_loop,)).start()

        queue_name = f"{self.model_name_hash}"

        async def recv_from_nats():
            nats_client = await nats.connect(self.nats_endpoint)
            js = nats_client.jetstream()

            await js.add_stream(
                name=queue_name,
                subjects=[queue_name],
                max_age=60 * 60 * 24,  # 1 day
                max_bytes=1024 * 1024 * 1024,  # 1 GB
                max_msgs=1000000,  # 1 million messages
            )

            sub = await js.subscribe(queue_name, queue=queue_name)

            while True:
                msg = await sub.next_msg(timeout=None)
                await msg.ack()
                remote_prefill_req: RemotePrefillReq = pickle.loads(msg.data)
                logger.debug(
                    f"Recv request {remote_prefill_req.rid} and bootstrap room: {remote_prefill_req.bootstrap_room}."
                )
                self.remote_prefill_reqs.append(remote_prefill_req)

        asyncio.run_coroutine_threadsafe(recv_from_nats(), event_loop)
        logger.debug(f"Start thread to recv queue succeed!")
