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

import hashlib
import logging
import time
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.hidden_state import (
    get_pd_hidden_capture_layer_ids,
    get_pd_hidden_req_state as pd_hidden_state,
)
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_dsa_seed_metadata_dim,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_aborted,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
    resolve_disagg_metadata_config,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.speculative.dspark_components.dspark_disaggregation import (
    resolve_hidden_bootstrap_plan,
)
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.prefill_attempt_count > 1 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def clear_pd_hidden_request_state(req: Req) -> None:
    pd_hidden_state(req).meta = None
    pd_hidden_state(req).src_indices = None
    pd_hidden_state(req).dst_indices = None
    pd_hidden_state(req).written = None
    pd_hidden_state(req).capture_layer_ids = None
    pd_hidden_state(req).current_src_indices = None
    pd_hidden_state(req).current_start = None
    pd_hidden_state(req).current_row_len = 0
    pd_hidden_state(req).current_is_last = False
    pd_hidden_state(req).owner_direct_sent = False


def maybe_release_metadata_buffer(
    req: Req,
    allocator: ReqToMetadataIdxAllocator,
    pd_hidden_pool=None,
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1
    indices = pd_hidden_state(req).src_indices
    if indices:
        sender = req.disagg_kv_sender
        if pd_hidden_pool is None and sender is not None:
            pd_hidden_pool = sender.kv_mgr.pd_hidden_pool
        worker_released = (
            sender is not None
            and sender.kv_mgr.pop_pd_hidden_request_done(sender.bootstrap_room)
        )
        if not worker_released and pd_hidden_pool is not None:
            pd_hidden_pool.free(indices)
        clear_pd_hidden_request_state(req)
    elif not indices:
        clear_pd_hidden_request_state(req)


def maybe_release_pd_hidden_rows(req: Req, pd_hidden_pool) -> None:
    """Release source hidden rows once the local RDMA transfer is complete."""
    if pd_hidden_pool is None:
        return
    indices = pd_hidden_state(req).src_indices
    if indices:
        pd_hidden_pool.free(indices)
        clear_pd_hidden_request_state(req)


def maybe_release_pd_hidden_rows_on_hidden_done(
    req: Req, pd_hidden_pool
) -> bool:
    """Release source hidden rows after PD_HIDDEN finishes, before KV success."""
    indices = pd_hidden_state(req).src_indices
    if not indices or pd_hidden_pool is None:
        return False
    sender = req.disagg_kv_sender
    if sender is None or not sender.kv_mgr.pop_pd_hidden_request_done(
        sender.bootstrap_room
    ):
        return False

    clear_pd_hidden_request_state(req)
    return True


def fail_pd_hidden_transfer(req: Req, message: str) -> None:
    """Route a PD hidden transfer failure through the standard KV failed path."""
    logger.warning(message)
    sender = req.disagg_kv_sender
    if sender is None:
        return
    kv_mgr = sender.kv_mgr
    room = sender.bootstrap_room
    kv_mgr.record_failure(room, message)
    kv_mgr.update_status(room, KVPoll.Failed)
    kv_mgr._wake_pd_hidden_ack_waiters(room)
    sender.conclude_state = KVPoll.Failed


def is_pd_hidden_transfer_failed(req: Req) -> bool:
    sender = req.disagg_kv_sender
    return sender is not None and sender.conclude_state == KVPoll.Failed


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
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.scheduler = scheduler
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.effective_max_total_num_tokens
        )
        self._last_pd_hidden_credit_warning_time = 0.0
        self.transfer_backend = transfer_backend
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get() and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        layer_shard_rank = getattr(self.token_to_kv_pool, "layer_shard_rank", None)
        layer_shard_size = getattr(self.token_to_kv_pool, "layer_shard_size", 1)
        transfer_draft_cache = (
            not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1
        )
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else self.token_to_kv_pool.start_layer
        )
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

        if self.draft_token_to_kv_pool is not None and transfer_draft_cache:
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
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
            pd_hidden_pool=getattr(self.metadata_buffers, "pd_hidden_pool", None),
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # V4's KVCache is organized by compression-ratio
            # buckets rather than by layer.
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
            )

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        kv_manager.pd_hidden_pool = getattr(self.metadata_buffers, "pd_hidden_pool", None)
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
                )
        return kv_manager

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity."""
        if self._check_if_req_exceed_kv_capacity(req):
            return False

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
            req_has_disagg_prefill_dp_rank=req.disagg_prefill_dp_rank is not None,
        )
        self._process_req(req)
        req.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if req.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        assert req.metadata_buffer_index is not None
        return True

    def _requires_pd_hidden_transfer(self, req: Req) -> bool:
        if self.kv_manager.req_to_pd_hidden_meta.get(req.bootstrap_room):
            return True
        return StateType.PD_HIDDEN in self.kv_manager.kv_args.state_types

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        metadata_buffer_was_unallocated = req.metadata_buffer_index < 0
        if not self.ensure_metadata_buffer(req):
            return False

        decode_prefix_len = getattr(req, "disagg_decode_prefix_len", None)
        if decode_prefix_len is None:
            decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
            req.disagg_decode_prefix_len = decode_prefix_len
        dspark_meta = self.kv_manager.req_to_pd_hidden_meta.get(req.bootstrap_room)
        if dspark_meta and not self._finalize_pd_hidden_bootstrap(
            req, dspark_meta, decode_prefix_len
        ):
            if metadata_buffer_was_unallocated and req.metadata_buffer_index >= 0:
                self.req_to_metadata_buffer_idx_allocator.free(
                    req.metadata_buffer_index
                )
                req.metadata_buffer_index = -1
            return False

        req.time_stats.set_bootstrap_done_time()
        num_kv_indices = len(req.origin_input_ids)
        req.start_send_idx = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        num_pages = kv_to_page_num(
            num_kv_indices_to_send, self.token_to_kv_pool.page_size
        )
        req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
        req.pending_bootstrap = False
        return True

    def _probe_bootstrap_ready(
        self,
        req: Req,
        metadata_credits: int,
        hidden_row_credits: int,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        """Validate metadata readiness without reserving transfer resources."""
        metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
        if metadata_cost > metadata_credits:
            return None, None

        dspark_meta = self.kv_manager.req_to_pd_hidden_meta.get(req.bootstrap_room)
        if not dspark_meta:
            return (metadata_cost, 0), None

        decode_prefix_len = getattr(req, "disagg_decode_prefix_len", None)
        if decode_prefix_len is None:
            decode_prefix_len = self.kv_manager.req_to_decode_prefix_len.get(
                req.bootstrap_room
            )
        if decode_prefix_len is None:
            return None, None

        plan, error = resolve_hidden_bootstrap_plan(
            req=req,
            metadata=dspark_meta,
            decode_prefix_len=decode_prefix_len,
            pp_rank=self.pp_rank,
            model_config=self.scheduler.model_config,
            model_runner=self.scheduler.tp_worker.model_runner,
            metadata_buffers=self.metadata_buffers,
            prefill_radix_enabled=not bool(
                self.scheduler.server_args.disable_radix_cache
            ),
        )
        if error is not None:
            return None, error
        assert plan is not None
        if not plan.local_layer_ids:
            return (metadata_cost, 0), None

        hidden_cost = 0 if plan.streaming_hidden else plan.source_window_rows
        if pd_hidden_state(req).src_indices is not None:
            hidden_cost = 0
        if hidden_cost > hidden_row_credits:
            now = time.monotonic()
            if now - self._last_pd_hidden_credit_warning_time > 30:
                logger.warning(
                    "PD hidden pool blocked prefill bootstrap: "
                    "rid=%s hidden_len=%d required_rows=%d free_rows=%d "
                    "pool_rows=%d bootstrap_queue=%d",
                    req.rid,
                    plan.hidden_len,
                    hidden_cost,
                    hidden_row_credits,
                    plan.pool.size,
                    len(self.queue),
                )
                self._last_pd_hidden_credit_warning_time = now
            return None, None

        return (metadata_cost, hidden_cost), None

    def _is_pd_hidden_credit_blocked(
        self, req: Req, metadata_credits: int, hidden_row_credits: int
    ) -> bool:
        metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
        if metadata_cost > metadata_credits:
            return False
        dspark_meta = self.kv_manager.req_to_pd_hidden_meta.get(req.bootstrap_room)
        if not dspark_meta:
            return False

        pp_slices = dspark_meta.get("pp_slices") or {}
        local_pp_slice = pp_slices.get(str(self.pp_rank)) if pp_slices else None
        local_layer_ids = (
            [int(x) for x in local_pp_slice.get("layer_ids", [])]
            if local_pp_slice
            else (
                []
                if pp_slices
                else [int(x) for x in dspark_meta.get("target_layer_ids", [])]
            )
        )
        if not local_layer_ids or pd_hidden_state(req).src_indices:
            return False

        pool = getattr(self.metadata_buffers, "pd_hidden_pool", None)
        if pool is None:
            return False
        hidden_len = int(dspark_meta.get("hidden_len", len(req.origin_input_ids)))
        streaming_hidden = bool(dspark_meta.get("streaming_hidden", False))
        window_rows = int(dspark_meta.get("streaming_window_rows", hidden_len))
        required_rows = 0 if streaming_hidden else min(hidden_len, window_rows)
        return required_rows <= pool.size and required_rows > hidden_row_credits

    def stage_pp_bootstrap_consensus(self, rids: List[str]) -> List[str]:
        """Enter the resource-commit phase after metadata consensus."""
        rid_set = set(rids)
        committed = []
        for req in self.queue:
            if req.rid not in rid_set:
                continue
            req.dspark_pp_bootstrap_consensus = True
            if req.pending_bootstrap and not should_force_retry(req):
                if self.finalize_bootstrap(req):
                    committed.append(req.rid)
        return committed

    def _abort_pd_hidden_bootstrap(self, req: Req, message: str) -> None:
        logger.error(message)
        prepare_abort(req, message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        sender = req.disagg_kv_sender
        if sender is not None:
            sender.kv_mgr.record_failure(sender.bootstrap_room, message)
            sender.kv_mgr.update_status(sender.bootstrap_room, KVPoll.Failed)
            sender.conclude_state = KVPoll.Failed

    def _finalize_pd_hidden_bootstrap(
        self, req: Req, dspark_meta: dict, decode_prefix_len: int
    ) -> bool:
        plan, error = resolve_hidden_bootstrap_plan(
            req=req,
            metadata=dspark_meta,
            decode_prefix_len=decode_prefix_len,
            pp_rank=self.pp_rank,
            model_config=self.scheduler.model_config,
            model_runner=self.scheduler.tp_worker.model_runner,
            metadata_buffers=self.metadata_buffers,
            prefill_radix_enabled=not bool(
                self.scheduler.server_args.disable_radix_cache
            ),
        )
        if error is not None:
            self._abort_pd_hidden_bootstrap(req, error)
            return False

        assert plan is not None
        if not plan.local_layer_ids:
            pd_hidden_state(req).meta = dict(dspark_meta)
            pd_hidden_state(req).src_indices = []
            pd_hidden_state(req).dst_indices = []
            pd_hidden_state(req).written = []
            pd_hidden_state(req).owner_direct_sent = False
            return True

        src_indices = (
            None
            if plan.streaming_hidden
            else plan.pool.alloc(plan.source_window_rows)
        )
        if src_indices is None and not plan.streaming_hidden:
            message = (
                "PD hidden rows exceed prefill hidden pool capacity: "
                f"rid={req.rid}, hidden_len={plan.hidden_len}, "
                f"required_rows={plan.source_window_rows}, "
                f"pool_size={plan.pool.size}. "
                "Increase SGLANG_PD_HIDDEN_POOL_TOKENS or reduce the "
                "maximum prompt/hidden transfer length."
            )
            self._abort_pd_hidden_bootstrap(req, message)
            return False

        pd_hidden_state(req).capture_layer_ids = [int(x) for x in plan.local_layer_ids]
        pd_hidden_state(req).meta = dict(dspark_meta)
        pd_hidden_state(req).src_indices = src_indices
        pd_hidden_state(req).dst_indices = plan.dst_indices
        pd_hidden_state(req).written = (
            None if plan.streaming_hidden else [False] * plan.hidden_len
        )
        pd_hidden_state(req).owner_direct_sent = False
        return True

    def add(self, req: Req, num_kv_heads: int) -> None:
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
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
        rids_to_check_set = set(rids_to_check) if rids_to_check is not None else None

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if (
                rids_to_check_set is not None
                and req.rid not in rids_to_check_set
                and poll != KVPoll.Failed
            ):
                # In PP mode, successful bootstrap still requires cross-rank
                # consensus. Local failures are terminal and must be drained
                # even if an earlier PP rank has already removed the request.
                continue

            if poll == KVPoll.Failed:
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif poll == KVPoll.Bootstrapping:
                if self._requires_pd_hidden_transfer(req):
                    # PD hidden must be captured for every prefill chunk.
                    # Do not run optimistic forward before hidden rows and
                    # capture metadata are materialized.
                    continue
                if (
                    req.prefill_attempt_count
                    < self.scheduler.server_args.optimistic_prefill_attempts
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                elif req.pending_bootstrap and not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def get_ready_bootstrapped_rids_for_pp(self) -> Tuple[List[str], List[str]]:
        """Return ordered PP candidates using a side-effect-free credit probe."""
        good_rids: List[str] = []
        failed_rids: List[str] = []
        if len(self.queue) == 0:
            return good_rids, failed_rids

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )

        metadata_credits = (
            self.req_to_metadata_buffer_idx_allocator.available_size()
        )
        pool = getattr(self.metadata_buffers, "pd_hidden_pool", None)
        hidden_row_credits = pool.available_size() if pool is not None else 0

        for req, poll in zip(self.queue, polls):
            if poll == KVPoll.Failed:
                failed_rids.append(req.rid)
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):
                    metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
                    if metadata_cost > metadata_credits:
                        break
                    metadata_credits -= metadata_cost
                elif req.pending_bootstrap:
                    costs, error = self._probe_bootstrap_ready(
                        req, metadata_credits, hidden_row_credits
                    )
                    if error is not None:
                        self._abort_pd_hidden_bootstrap(req, error)
                        failed_rids.append(req.rid)
                        continue
                    if costs is None:
                        if self._is_pd_hidden_credit_blocked(
                            req, metadata_credits, hidden_row_credits
                        ):
                            break
                        break
                    metadata_cost, hidden_cost = costs
                    metadata_credits -= metadata_cost
                    hidden_row_credits -= hidden_cost
                good_rids.append(req.rid)
            elif poll == KVPoll.Bootstrapping:
                continue
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} "
                    "in get_ready_bootstrapped_rids_for_pp"
                )
        return good_rids, failed_rids

    def release_memory_occupation(self):
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def init_disaggregation(self: Scheduler) -> None:
        from sglang.srt.configs.model_config import is_minimax_sparse
        from sglang.srt.disaggregation.decode import (
            DecodePreallocQueue,
            DecodeTransferQueue,
        )
        from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
        from sglang.srt.mem_cache import kv_cache_builder
        from sglang.srt.speculative.eagle_utils import (
            get_draft_recurrent_hidden_state_spec,
        )

        self.mm_receiver = None
        self.disagg_prefill_bootstrap_queue = None
        self.disagg_prefill_inflight_queue = None
        self.disagg_decode_prealloc_queue = None
        self.disagg_decode_transfer_queue = None

        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
        draft_token_to_kv_pool = kv_cache_builder.get_draft_kv_pool(
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
        )

        if self.spec_algorithm.carries_draft_hidden_states():
            # `draft_runner` aliases `draft_runner_list[0]` in the multi-layer
            # worker, so a single accessor covers both shapes.
            draft_runner = self.draft_worker.draft_worker.draft_runner
            disagg_hidden_size, disagg_hidden_states_dtype = (
                get_draft_recurrent_hidden_state_spec(draft_runner)
            )
        else:
            disagg_hidden_size = 16  # minimal padding size for RDMA
            disagg_hidden_states_dtype = torch.float32

        disagg_metadata_config = resolve_disagg_metadata_config(
            hidden_size=disagg_hidden_size,
            hidden_states_dtype=disagg_hidden_states_dtype,
            disaggregation_mode=self.disaggregation_mode,
            transfer_backend=self.transfer_backend,
            spec_algorithm=self.spec_algorithm,
            model_config=self.model_config,
            server_args=self.server_args,
            model_runner=self.tp_worker.model_runner,
            pp_rank=self.ps.pp_rank,
            pp_size=self.ps.pp_size,
            gpu_id=self.ps.gpu_id,
            max_prefill_tokens=self.max_prefill_tokens,
        )
        disagg_hidden_size = disagg_metadata_config.hidden_size
        disagg_hidden_states_dtype = disagg_metadata_config.hidden_states_dtype
        metadata_buffer_kwargs = disagg_metadata_config.metadata_buffer_kwargs

        # The PD metadata wire schema must match on P and D even when only D
        # enables spec decoding; a seedless prefill writes the invalid sentinel.
        output_dsa_topk_indices_dim = get_dsa_seed_metadata_dim(
            self.model_config.hf_config
        )

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *8 headroom for MiniMax-M3; *2 for other models.
            buffer_multiplier = (
                8 if is_minimax_sparse(self.model_config.hf_config) else 2
            )
            buffer_size = (self.req_to_token_pool.size) * buffer_multiplier
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
                output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
                **metadata_buffer_kwargs,
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.ps.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.ps.pp_rank,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
                output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
                **metadata_buffer_kwargs,
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.ps.pp_rank,
                pp_size=self.ps.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

            self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        # Init mm receiver for EPD disaggregation mode
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                pp_rank=self.ps.pp_rank,
                tp_rank=self.ps.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.
        """
        candidates = [req for req in self.waiting_queue if not is_aborted(req)]
        if not candidates:
            return
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in candidates],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        failed = set()
        for req, poll in zip(candidates, polls):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        if failed:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in failed
            ]

    def has_bootstrapped_waiting_req(self: Scheduler) -> bool:
        return any(
            not req.pending_bootstrap and not is_aborted(req)
            for req in self.waiting_queue
        )

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
        running_batch: ScheduleBatch,
        last_batch: Optional[ScheduleBatch],
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        running_batch.batch_is_full = False

        self.resolve_waiting_queue_bootstrap()

        self.process_prefill_chunk(last_batch=last_batch, running_batch=running_batch)

        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            self._apply_war_barrier()

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

    def _extract_pd_hidden_states_from_result(
        self: Scheduler,
        result: GenerationBatchResult,
    ) -> Optional[torch.Tensor]:
        logits_output = result.logits_output
        hidden_states = getattr(logits_output, "hidden_states", None)
        if hidden_states is None and result.pp_hidden_states_proxy_tensors is not None:
            proxy_tensors = result.pp_hidden_states_proxy_tensors.tensors
            aux_keys = sorted(
                key
                for key in proxy_tensors
                if key.startswith("pd_aux_hidden_states_")
            )
            if aux_keys:
                hidden_states = (
                    proxy_tensors[aux_keys[0]]
                    if len(aux_keys) == 1
                    else torch.cat([proxy_tensors[key] for key in aux_keys], dim=-1)
                )
        return hidden_states

    def _build_pd_hidden_only_state_indices(
        self: Scheduler, req: Req
    ) -> Optional[List]:
        current_indices = pd_hidden_state(req).current_src_indices
        if current_indices is None:
            return None

        state_types = (
            self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
        )
        state_indices = []
        for st in state_types:
            if st == StateType.PD_HIDDEN:
                state_indices.append(np.asarray(current_indices, dtype=np.int32))
            else:
                state_indices.append(None)
        return state_indices

    def _send_pd_hidden_only_chunk(self: Scheduler, req: Req) -> bool:
        current_indices = pd_hidden_state(req).current_src_indices
        current_start = pd_hidden_state(req).current_start
        current_rows = int(pd_hidden_state(req).current_row_len or 0)
        if current_indices is None or current_start is None or current_rows <= 0:
            return False

        state_indices = self._build_pd_hidden_only_state_indices(req)
        if state_indices is None:
            return False

        streaming_hidden = bool(
            (pd_hidden_state(req).meta or {}).get("streaming_hidden", False)
        )
        if req.disagg_kv_sender is not None:
            source_event = self.device_module.Event()
            source_event.record()
            req.disagg_kv_sender.set_source_event(source_event)
            req.disagg_kv_sender.set_pd_hidden_chunk_meta(
                int(current_start),
                int(current_rows),
                bool(pd_hidden_state(req).current_is_last),
                current_indices if streaming_hidden else pd_hidden_state(req).src_indices,
            )

        req.disagg_kv_sender.send(np.asarray([], dtype=np.int32), state_indices)
        if streaming_hidden:
            pd_hidden_state(req).src_indices = None
        pd_hidden_state(req).current_src_indices = None
        pd_hidden_state(req).current_start = None
        pd_hidden_state(req).current_row_len = 0
        pd_hidden_state(req).current_is_last = False
        pd_hidden_state(req).owner_direct_sent = True
        return True

    def _write_pd_hidden_rows_for_batch(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        *,
        send_owner_direct: bool = False,
    ) -> None:
        pool = getattr(self.disagg_metadata_buffers, "pd_hidden_pool", None)
        hidden_states = self._extract_pd_hidden_states_from_result(result)
        needs_pd_hidden_reqs = [
            req
            for req in batch.reqs
            if (
                (
                    pd_hidden_state(req).src_indices
                    or pd_hidden_state(req).capture_layer_ids
                )
                and (
                    send_owner_direct
                    or not pd_hidden_state(req).owner_direct_sent
                )
            )
        ]
        if pool is not None and needs_pd_hidden_reqs and hidden_states is None:
            reqs = [
                (
                    req.rid,
                    pd_hidden_state(req).capture_layer_ids,
                    bool(pd_hidden_state(req).src_indices),
                )
                for req in needs_pd_hidden_reqs
            ]
            message = (
                "PD hidden capture was required but forward output has no "
                "hidden states: batch_capture_layers="
                f"{get_pd_hidden_capture_layer_ids(batch.reqs)}, "
                f"reqs={reqs}"
            )
            for req in needs_pd_hidden_reqs:
                fail_pd_hidden_transfer(req, message)
            return
        if pool is None or hidden_states is None or batch.extend_lens is None:
            return

        if batch.seq_lens_cpu is not None:
            chunk_ends = [int(x) for x in batch.seq_lens_cpu.tolist()]
        else:
            assert batch.prefix_lens is not None
            chunk_ends = [
                int(prefix_len) + int(extend_len)
                for prefix_len, extend_len in zip(
                    batch.prefix_lens, batch.extend_lens, strict=True
                )
            ]

        hidden_offset = 0
        for req, extend_len, chunk_end in zip(
            batch.reqs, batch.extend_lens, chunk_ends, strict=True
        ):
            extend_len = int(extend_len)
            req_hidden = hidden_states[hidden_offset : hidden_offset + extend_len]
            hidden_offset += extend_len

            meta = pd_hidden_state(req).meta or {}
            streaming_hidden = bool(meta.get("streaming_hidden", False))
            if not send_owner_direct and pd_hidden_state(req).owner_direct_sent:
                continue
            src_indices = pd_hidden_state(req).src_indices
            if not src_indices and not streaming_hidden:
                continue

            hidden_start = int(meta.get("hidden_start", 0))
            hidden_len = int(meta.get("hidden_len", len(src_indices or [])))
            chunk_start = chunk_end - extend_len
            write_start = max(chunk_start, hidden_start)
            write_end = min(chunk_end, hidden_start + hidden_len)
            if write_end <= write_start:
                continue

            local_start = write_start - hidden_start
            local_end = write_end - hidden_start
            chunk_local_start = write_start - chunk_start
            chunk_local_end = write_end - chunk_start
            req_hidden_to_write = req_hidden
            pp_slices = meta.get("pp_slices") or {}
            pp_rank = int(self.ps.pp_rank)
            local_pp_slice = pp_slices.get(str(pp_rank)) if pp_slices else None
            local_slice_len = (
                int(local_pp_slice.get("slice_len", 0))
                if local_pp_slice
                else pool.hidden_size
            )
            if local_slice_len > 0 and req_hidden_to_write.shape[-1] != local_slice_len:
                local_slice_start = (
                    int(local_pp_slice.get("slice_start", 0))
                    if local_pp_slice
                    else 0
                )
                local_slice_end = local_slice_start + local_slice_len
                if req_hidden_to_write.shape[-1] < local_slice_end:
                    raise RuntimeError(
                        "PD hidden width does not match prefill PP slice: "
                        f"rid={req.rid}, pp_rank={pp_rank}, "
                        f"hidden_width={req_hidden_to_write.shape[-1]}, "
                        f"slice_start={local_slice_start}, "
                        f"slice_len={local_slice_len}"
                    )
                req_hidden_to_write = req_hidden_to_write[
                    :, local_slice_start:local_slice_end
                ]
            if streaming_hidden:
                rows = local_end - local_start
            else:
                rows = local_end - local_start
                write_indices = src_indices[local_start:local_end]
            prev_current_start = pd_hidden_state(req).current_start
            prev_current_row_len = int(pd_hidden_state(req).current_row_len or 0)
            if (
                prev_current_start is not None
                and prev_current_row_len > 0
                and int(prev_current_start) != int(write_start)
            ):
                if req.pending_bootstrap:
                    raise RuntimeError(
                        "PD streaming hidden current chunk would be overwritten "
                        "before bootstrap is finalized: "
                        f"rid={req.rid}, old_start={prev_current_start}, "
                        f"old_rows={prev_current_row_len}, new_start={write_start}, "
                        f"new_rows={rows}"
                    )
                self.send_kv_chunk(
                    req,
                    last_chunk=False,
                    end_idx=int(prev_current_start) + prev_current_row_len,
                )
            if streaming_hidden:
                write_indices = pool.alloc(rows)
                if write_indices is None:
                    fail_pd_hidden_transfer(
                        req,
                        "PD streaming hidden source chunk allocation failed: "
                        f"rid={req.rid}, rows={rows}, free_rows={pool.available_size()}, "
                        f"pool_rows={pool.size}. Streaming source rows are released "
                        "only after the matching hidden chunk ACK.",
                    )
                    continue
                pd_hidden_state(req).src_indices = write_indices
            pool.write(
                write_indices,
                req_hidden_to_write[chunk_local_start:chunk_local_end],
            )
            pd_hidden_state(req).current_start = write_start
            pd_hidden_state(req).current_row_len = rows
            pd_hidden_state(req).current_src_indices = write_indices
            pd_hidden_state(req).current_is_last = write_end >= hidden_start + hidden_len
            written = pd_hidden_state(req).written
            if written is not None:
                written[local_start:local_end] = [True] * rows
            if send_owner_direct:
                self._send_pd_hidden_only_chunk(req)

    def send_dspark_owner_direct_hidden_for_batch(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> bool:
        capture_reqs = [
            req
            for req in batch.reqs
            if pd_hidden_state(req).capture_layer_ids
        ]
        if not capture_reqs:
            return False
        if any(req.pending_bootstrap for req in capture_reqs):
            return False
        if not all(
            bool(
                (pd_hidden_state(req).meta or {}).get("streaming_hidden", False)
            )
            for req in capture_reqs
        ):
            return False
        self._write_pd_hidden_rows_for_batch(
            batch, result, send_owner_direct=True
        )
        return True

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
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
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logprob_pt = 0
        assert batch.spec_info is result.next_draft_input
        draft_input = result.next_draft_input
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )
        self._write_pd_hidden_rows_for_batch(batch, result)

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                # Test hook: exercise the release/requeue retry path.
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue

                if is_aborted(req) or is_pd_hidden_transfer_failed(req):
                    self.disagg_prefill_inflight_queue.append(req)
                    req.time_stats.set_prefill_transfer_queue_entry_time()
                    advance_logprob_pt(i, req)
                    continue

                req.output_ids.append(next_token_id)
                maybe_cache_unfinished_req(req, self.tree_cache)
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and draft_input is not None:
                    req.output_topk_p = draft_input.topk_p[i]
                    req.output_topk_index = draft_input.topk_index[i]
                    req.hidden_states_tensor = (
                        draft_input.hidden_states[i].cpu().clone()
                    )
                    dsa_topk_indices = batch.spec_info.dsa_topk_indices
                    if dsa_topk_indices is not None:
                        req.output_dsa_topk_indices = dsa_topk_indices[i].cpu().clone()
                    else:
                        req.output_dsa_topk_indices = None
                else:
                    req.hidden_states_tensor = None
                    req.output_dsa_topk_indices = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                if req.return_sampling_mask:
                    self.batch_result_processor.add_sampling_mask_return_values(
                        i, req, logits_output
                    )
                if not req.pending_bootstrap:
                    self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()

                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1

                # Still chunking iff its next chunk was launched: either it is
                # still self.chunked_req, or its final chunk (extend_range
                # reaching the end of the input) is in flight. A yielded req
                # is neither, so do its deferred release here.
                still_chunking = self.chunked_req is req or (
                    req.extend_range is not None
                    and req.extend_range.end >= len(req.origin_input_ids)
                )
                if req.pending_bootstrap and not still_chunking:
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # Optimistic bootstrap can fail while this overlapped chunk is
                # already running. Drop aborted chunks instead of sending KV.
                if is_aborted(req):
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                # In non-overlap-mode, KV is sent in process_prefill_chunk
                # Only send when req's sender is initialized
                if self.enable_overlap and not req.pending_bootstrap:
                    assert (
                        req.metadata_buffer_index >= 0
                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

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

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        terminal_rids_to_check = set(rids_to_check) if rids_to_check is not None else None
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if terminal_rids_to_check is not None:
                if req.rid not in terminal_rids_to_check:
                    undone_reqs.append(req)
                    continue

                # In PP mode, the previous rank may have reached a terminal
                # state (Success/Failed) while this rank's local poll is still
                # in a transient state due to clock skew or propagation delay.
                # Treat non-terminal states as undone instead of crashing.
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning_once(
                        f"PP rank {self.ps.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone",
                    )
                    undone_reqs.append(req)
                    continue

            maybe_release_pd_hidden_rows_on_hidden_done(
                req,
                getattr(self.disagg_metadata_buffers, "pd_hidden_pool", None),
            )

            if req.pending_bootstrap:
                # Parked: prefill finished before bootstrap completed.
                if self.handle_pending_bootstrap(req, poll):
                    self.send_kv_chunk(req, last_chunk=True)
                    undone_reqs.append(req)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # todo: set Transferring correctly in backend
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                self.handle_inflight_transfer_failure(req)
                done_reqs.append(req)
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in inflight queue"
                )

        for req in done_reqs:
            req.time_stats.set_completion_time()

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.output_streamer.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req,
                self.req_to_metadata_buffer_idx_allocator,
                getattr(self.disagg_metadata_buffers, "pd_hidden_pool", None),
            )

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def handle_inflight_transfer_failure(
        self: Scheduler, req: Req
    ) -> Optional[Exception]:
        """Conclude an inflight request whose KV transfer failed."""
        error_message = (
            f"Prefill transfer failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        exc: Optional[Exception] = None
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            exc = e
            error_message += f" with exception {e}"
        # Mute error message for propagated exceptions to avoid duplicate logging
        if getattr(exc, "is_from_another_rank", False):
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        release_kv_cache(req, self.tree_cache)  # unlock the tree
        if not isinstance(req.finished_reason, FINISH_ABORT):
            prepare_abort(
                req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()
        return exc

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP to inspect local terminal transfers without popping requests.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        transferred_rids: List[str] = []
        pd_hidden_pool = getattr(
            self.disagg_metadata_buffers, "pd_hidden_pool", None
        )

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            maybe_release_pd_hidden_rows_on_hidden_done(req, pd_hidden_pool)
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # Mute error message for propagated exceptions to avoid duplicate logging
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if (
            req.req_pool_idx is not None
            or req.kv is not None
            or req.mamba_pool_idx is not None
        ):
            release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(
            req,
            self.req_to_metadata_buffer_idx_allocator,
            getattr(self.disagg_metadata_buffers, "pd_hidden_pool", None),
        )
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)

    def handle_pending_bootstrap(self: Scheduler, req: Req, poll: KVPoll) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            return False
        elif poll == KVPoll.WaitingForInput:
            if should_force_retry(req):  # test hook
                return False
            if self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req):
                return True
            if is_aborted(req):
                self.handle_bootstrap_failure(req)
            return False
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self: Scheduler, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(req, polls[0])

    def process_prefill_chunk(
        self: Scheduler,
        last_batch: Optional[ScheduleBatch],
        running_batch: ScheduleBatch,
    ) -> None:
        chunked_req_to_exclude = set()
        if (req := self.chunked_req) is not None:
            chunked_req_to_exclude.add(req)
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

            if not self.check_bootstrap(req):
                if is_aborted(req):
                    # bootstrap failed
                    self.chunked_req = None
                elif self.disagg_prefill_bootstrap_queue._requires_pd_hidden_transfer(
                    req
                ):
                    self.chunked_req = None
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                elif self.has_bootstrapped_waiting_req():
                    # optimistic request yields to waiting requests
                    self.chunked_req = None
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                # else: still bootstrapping, keep computing without sending
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                req.tmp_end_idx = min(
                    req.extend_range.end,
                    len(req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(req)

            if self.chunked_req is not None:
                running_batch.batch_is_full = False

        if last_batch and last_batch.forward_mode.is_extend():
            if last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(last_batch.chunked_req)

            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

    def maybe_send_cached_prefix_chunk(self: Scheduler, req: Req) -> None:
        # Only bootstrap-finalized requests; staging excluded.
        if (
            not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get()
            or self.enable_staging
            or req.pending_bootstrap
        ):
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = len(req.prefix_indices) - req.host_hit_length
        if cached_end <= req.start_send_idx:
            return
        assert cached_end % self.token_to_kv_pool_allocator.page_size == 0
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> bool:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, transfer_input_len)
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return True

        current_pd_hidden_src_indices = pd_hidden_state(req).current_src_indices
        current_pd_hidden_start = pd_hidden_state(req).current_start
        current_pd_hidden_row_len = int(pd_hidden_state(req).current_row_len or 0)
        has_current_pd_hidden = (
            current_pd_hidden_src_indices is not None
            and current_pd_hidden_row_len > 0
        )
        streaming_pd_hidden = bool(
            (pd_hidden_state(req).meta or {}).get("streaming_hidden", False)
        )

        state_indices: Optional[List] = None
        if last_chunk or has_current_pd_hidden:
            if last_chunk:
                self.disagg_metadata_buffers.set_buf(req)

            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # that decode used to register the destination row.
            seq_len = min(end_idx, transfer_input_len)
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _dsa_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                online = is_dsv4_c128_online_enabled()
                ring_size = (
                    1
                    if online
                    else self.token_to_kv_pool_allocator.get_kvcache().get_ring_size(
                        128
                    )
                )
                return get_dsv4_c128_state_indices(
                    int(req.req_pool_idx),
                    c128_seq_len,
                    online=online,
                    ring_size=ring_size,
                )

            def _pd_hidden_payload():
                if pd_hidden_state(req).owner_direct_sent:
                    return []
                src_indices = pd_hidden_state(req).src_indices
                if src_indices is None and pd_hidden_state(req).capture_layer_ids:
                    raise RuntimeError(
                        "PD hidden row pool was not materialized before transfer: "
                        f"rid={req.rid}"
                    )
                if has_current_pd_hidden:
                    return np.asarray(current_pd_hidden_src_indices, dtype=np.int32)
                if not src_indices:
                    return []
                written = pd_hidden_state(req).written
                if written is not None and not all(written):
                    missing = [i for i, ok in enumerate(written) if not ok][:8]
                    raise RuntimeError(
                        "PD hidden rows are incomplete before transfer: "
                        f"rid={req.rid}, missing_offsets={missing}"
                    )
                return np.asarray(src_indices, dtype=np.int32)

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            state_indices = []
            for st in state_types:
                if st == StateType.MAMBA:
                    state_indices.append(_mamba_payload())
                elif st == StateType.SWA:
                    state_indices.append(_swa_payload())
                elif st == StateType.DSA:
                    state_indices.append(_dsa_payload())
                elif st == StateType.MINIMAX_INDEX_K:
                    # Index rows live at the same loc as main KV on the same
                    # page_size, so reuse the full-seq page-ids.
                    state_indices.append(_dsa_payload())
                elif st == StateType.SWA_RING:
                    state_indices.append(_swa_ring_payload())
                elif st == StateType.C128_STATE:
                    state_indices.append(_c128_state_payload())
                elif st == StateType.PD_HIDDEN:
                    state_indices.append(_pd_hidden_payload())
                else:
                    state_indices.append(None)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, start_idx:end_idx
        ]
        page_indices = kv_to_page_indices(kv_indices, page_size)
        should_send_kv_chunk = req.disagg_kv_sender.should_send_kv_chunk(
            len(page_indices), last_chunk
        )
        if not should_send_kv_chunk and not has_current_pd_hidden:
            return True
        if has_current_pd_hidden:
            source_event = self.device_module.Event()
            source_event.record()
            req.disagg_kv_sender.set_source_event(source_event)
            req.disagg_kv_sender.set_pd_hidden_chunk_meta(
                int(current_pd_hidden_start),
                int(current_pd_hidden_row_len),
                bool(pd_hidden_state(req).current_is_last),
                current_pd_hidden_src_indices
                if streaming_pd_hidden
                else pd_hidden_state(req).src_indices,
            )
        req.disagg_kv_sender.send(page_indices, state_indices)
        if has_current_pd_hidden and streaming_pd_hidden:
            pd_hidden_state(req).src_indices = None
        pd_hidden_state(req).current_src_indices = None
        pd_hidden_state(req).current_start = None
        pd_hidden_state(req).current_row_len = 0
        pd_hidden_state(req).current_is_last = False
        req.start_send_idx = end_idx
        return True

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        max_attempts = self.server_args.optimistic_prefill_attempts
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        req.tmp_end_idx = -1
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.prefill_attempt_count >= max_attempts:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill attempts "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            req.prefill_attempt_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill yielded "
                f"({req.prefill_attempt_count}/{max_attempts} attempts used)"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
