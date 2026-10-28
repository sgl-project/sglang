"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward; a launched request sits in `prefilling` until its step commits
    c. The commit of its last chunk adds the request to `transfer`

3. `transfer`
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import hashlib
import logging
from array import array
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.base import BaseKVSender, KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.checksum import (
    KvChecksumComputer,
    is_health_check_req,
    page_indices_for_request,
    state_indices_for_request,
)
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_buffer import (
    compute_grid_segments,
    staging_grid_tokens,
)
from sglang.srt.disaggregation.minimax import minimax_index_transfer_pages
from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    build_kv_layer_ids,
    build_metadata_buffers,
    build_staging_slot_metadata,
    get_dsa_tail_state_indices,
    get_kv_class,
    get_kv_transfer_buf_infos,
    get_pd_kv_layout,
    get_qsa_pending_state_indices,
    is_aborted,
    is_mla_backend,
    is_unadmitted_reject,
    kv_layout_signature,
    poll_and_all_reduce_attn_cp_tp_group,
    poll_and_all_reduce_prefill_pp,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.request_lists import RequestList
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    checkpoint_kv_cache,
    kv_to_page_indices,
    kv_to_page_num,
    suffix_page_indices,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_PROCESS_QUEUE,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.request_lists import CoreLists
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )
    from sglang.srt.managers.scheduler_components.bundles import (
        KvCapacity,
        KvResources,
        SchedulerServices,
        Streams,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.engine_state import LaunchCounters
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.observability.metrics_collector import SchedulerStats
    from sglang.srt.server_args import ServerArgs
logger = logging.getLogger(__name__)

# TP=1 yield between polls of a pending result. An initial heuristic rather
# than a tuned value: long enough that the bootstrap, transfer and cache
# threads reliably win the GIL even when the input socket is empty, and far
# below the duration of any prefill forward, so it adds no visible latency.
PREFILL_INPUT_POLL_INTERVAL_S = 0.0001

_is_npu = is_npu()


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.disagg.prefill_attempt_count > 1 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def _uses_write_through_cache(cache: object) -> bool:
    return (
        isinstance(cache, UnifiedRadixCache)
        and cache.cache_controller is not None
        and cache.cache_controller.write_policy == "write_through"
    )


def _transfer_start_layer(*, pool, hf_text_config) -> int:
    # Hybrid pools count all layers in start_layer, but peer KV lists contain only
    # full-attention layers, so translate to a full-attention-relative offset.
    if not isinstance(pool, HybridLinearKVPool):
        return pool.start_layer
    return sum(
        1 for lid in hf_text_config.full_attention_layer_ids if lid < pool.start_layer
    )


def max_bootstrap_input_len(*, capacity: int, page_size: int, shard_size: int) -> int:
    """Bootstrap prompt limit; sharded admission never fits a longer prompt."""
    if shard_size == 1:
        return capacity
    # Strictest path, add_one_req_ignore_eos: page-rounded prompt, one page per
    # shard, 1 new token and 1 spare token must stay below capacity.
    return (capacity - shard_size * page_size - 3) // page_size * page_size


def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.disagg.metadata_buffer_index >= 0:
        allocator.free(req.disagg.metadata_buffer_index)
        req.disagg.metadata_buffer_index = -1


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
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        kv_shard_widening: int,
        scheduler_stage_metrics: SchedulerStageMetricsRecorder,
        transfer_backend: TransferBackend,
        *,
        model_config: ModelConfig,
        server_args: ServerArgs,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        page_size: int,
        kv_cache_dtype_str: str,
        rust_http_port: Optional[int],
        attn_cp_cpu_group: Any,
        attn_tp_cpu_group: Any,
        pp_group: Any,
        output_streamer: SchedulerOutputStreamer,
        counters: LaunchCounters,
        request_retirer: RequestRetirer,
        handle_bootstrap_failure: Callable[[Req], None],
    ):
        parallel = get_parallel()
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = parallel.tp_rank
        self.pp_rank = parallel.pp_rank
        self.pp_size = parallel.pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        # The KV sender of each PD prefill request, by rid, from the bootstrap
        # until the request leaves PD prefill.
        self.kv_senders: Dict[str, BaseKVSender] = {}
        self.scheduler_stage_metrics = scheduler_stage_metrics
        self.max_total_num_tokens = max_total_num_tokens
        self.max_input_len = max_bootstrap_input_len(
            capacity=max_total_num_tokens,
            page_size=page_size,
            shard_size=kv_shard_widening,
        )
        self.transfer_backend = transfer_backend
        self.model_config = model_config
        self.server_args = server_args
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.page_size = page_size
        self.kv_cache_dtype_str = kv_cache_dtype_str
        self.rust_http_port = rust_http_port
        self.attn_cp_cpu_group = attn_cp_cpu_group
        self.attn_tp_cpu_group = attn_tp_cpu_group
        self.pp_group = pp_group
        self.output_streamer = output_streamer
        self.counters = counters
        self.request_retirer = request_retirer
        self.handle_bootstrap_failure = handle_bootstrap_failure
        self.pp_poll_sync_work_list: List = []
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            if self.is_mla_backend:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                    "(e.g. GQA, MHA). MLA models should not set this flag."
                )
            page_size = self.page_size
            # Same source as send_kv_chunk's staging grid below, so validation
            # and the grid cannot disagree after a post-publish override.
            chunked_prefill_size = get_schedule().chunked_prefill_size
            cps = chunked_prefill_size or 8192
            # Staging slices each send into a fixed page-aligned grid, so an
            # unbounded (-1) or non-page-aligned chunk size has no valid grid.
            if cps <= 0 or cps % page_size != 0:
                raise RuntimeError(
                    f"SGLANG_DISAGG_STAGING_BUFFER requires a positive "
                    f"chunked_prefill_size that is a multiple of page_size "
                    f"({page_size}); got {chunked_prefill_size}."
                )
            if self.pp_size > 1 and self.transfer_backend != TransferBackend.MOONCAKE:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER with pp_size > 1 is only "
                    "supported by Mooncake."
                )
        self.kv_manager = self._init_kv_manager()
        self.kv_checksum_signature = 0
        if get_disagg().disaggregation_enable_kv_checksum:
            kv_args = self.kv_manager.kv_args
            self.kv_checksum_signature = kv_layout_signature(kv_args)
            self.kv_checksum_computer = KvChecksumComputer(
                device=torch.device(f"cuda:{get_device().gpu_id}"),
                kv_data_ptrs=kv_args.kv_data_ptrs,
                kv_item_lens=kv_args.kv_item_lens,
                state_data_ptrs=kv_args.state_data_ptrs,
                state_item_lens=kv_args.state_item_lens,
            )
        else:
            self.kv_checksum_computer = None

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = get_parallel().dp_rank
        kv_args.rust_http_port = self.rust_http_port
        kv_args.kv_cache_dtype_str = self.kv_cache_dtype_str
        kv_args.kv_layout = get_pd_kv_layout(self.token_to_kv_pool)
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
            else _transfer_start_layer(
                pool=self.token_to_kv_pool,
                hf_text_config=self.model_config.hf_text_config,
            )
        )
        kv_data_ptrs, kv_data_lens, kv_item_lens = get_kv_transfer_buf_infos(
            self.token_to_kv_pool
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

        draft_kv_pool = (
            self.draft_token_to_kv_pool
            if transfer_draft_cache
            and (not _is_npu or get_parallel().pp_group.is_last_rank)
            else None
        )
        num_draft_entries = 0
        if draft_kv_pool is not None:
            # Draft KV shares target virtual ids. Unified target KV is transferred
            # with physical ids, so it needs a separate draft index vector.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                draft_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            num_draft_entries = len(draft_kv_data_ptrs)

        dcp_remote_decode_layout = []
        if self.transfer_backend == TransferBackend.ASCEND:
            for pool in (self.token_to_kv_pool, draft_kv_pool):
                get_layout = getattr(pool, "get_dcp_remote_decode_layout", None)
                if get_layout is not None:
                    dcp_remote_decode_layout.extend(get_layout())

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.num_draft_entries = num_draft_entries
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.token_to_kv_pool,
            draft_token_to_kv_pool=draft_kv_pool,
            num_draft_entries=num_draft_entries,
            num_hidden_layers=self.model_config.num_hidden_layers,
        )
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = self.model_config.get_total_num_kv_heads()
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = get_disagg().disaggregation_ib_device
        kv_args.gpu_id = get_device().gpu_id

        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.model_config.num_hidden_layers,
            req_to_token_pool=self.req_to_token_pool,
        )

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager_kwargs = (
            {"dcp_remote_decode_layout": dcp_remote_decode_layout}
            if self.transfer_backend == TransferBackend.ASCEND
            else {}
        )
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.server_args,
            self.is_mla_backend,
            **kv_manager_kwargs,
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            staging_slots = build_staging_slot_metadata(
                kv_layer_ids=kv_args.kv_layer_ids,
                num_draft_entries=num_draft_entries,
                kv_pool=kv_pool,
                draft_kv_pool=draft_kv_pool,
            )
            if staging_slots is not None:
                k_buffers, v_buffers, slot_layer_ids = staging_slots
                kv_manager.set_kv_buffer_tensors(
                    k_buffers,
                    v_buffers,
                    kv_pool.page_size,
                    slot_layer_ids=slot_layer_ids,
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

        self.kv_senders[req.rid] = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            req_has_disagg_prefill_dp_rank=req.disagg_prefill_dp_rank is not None,
        )
        self._process_req(req)
        req.disagg.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if req.disagg.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.disagg.metadata_buffer_index = (
            self.req_to_metadata_buffer_idx_allocator.alloc()
        )
        assert req.disagg.metadata_buffer_index is not None
        return True

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        assert req.disagg.pending_bootstrap, "finalize_bootstrap is not idempotent"
        if not self.ensure_metadata_buffer(req):
            return False

        req.time_stats.set_bootstrap_done_time()
        decode_prefix_len = self.kv_senders[req.rid].pop_decode_prefix_len()
        req.disagg.start_send_idx = decode_prefix_len
        # Base of the staging chunk grid (suffix-relative send coordinates).
        req.disagg.decode_prefix_len = decode_prefix_len
        num_pages = self._num_pages_to_send(req, decode_prefix_len)
        self.kv_senders[req.rid].init(num_pages, req.disagg.metadata_buffer_index)
        req.disagg.pending_bootstrap = False
        return True

    def _num_pages_to_send(self, req: Req, decode_prefix_len: int) -> int:
        return kv_to_page_num(
            len(req.origin_input_ids) - decode_prefix_len,
            self.page_size,
        )

    def add(self, req: Req, num_kv_heads: int) -> None:
        # Rejected at intake: `set_finish_with_abort` left the verdict in
        # `to_finish`, which `finished()` does not read, and swapped the prompt
        # for a one-token stub. Bootstrapping it costs a handshake, a metadata
        # buffer and a forward pass before anything unwinds it.
        if is_unadmitted_reject(req):
            self.request_retirer.retire_unadmitted_request(req)
            return
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_input_len:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_input_len}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def can_prefill_before_bootstrap(self, req: Req) -> bool:
        """Check eligibility; pause and metadata admission stay in the shared queue."""
        return (
            req.disagg.prefill_attempt_count < get_disagg().optimistic_prefill_attempts
            or get_disagg().disaggregation_decode_allocation_policy
            == "prefill_complete"
        ) and not (
            _uses_write_through_cache(self.tree_cache)
            and req.swa_branching_seqlen is not None
            and req.swa_branching_seqlen > req.kv.cache_protected_len
        )

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def pop_bootstrapped(self) -> List[Req]:
        """Pop the reqs which have finished bootstrapping."""

        bootstrapped_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            return []

        if self.pp_size > 1:
            polls = poll_and_all_reduce_prefill_pp(
                [self.kv_senders[req.rid] for req in self.queue],
                self.attn_cp_cpu_group,
                self.attn_tp_cpu_group,
                self.pp_group,
                self.pp_rank,
                self.pp_size,
                self.pp_poll_sync_work_list,
            )
        else:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [self.kv_senders[req.rid] for req in self.queue],
                self.attn_cp_cpu_group,
                self.attn_tp_cpu_group,
            )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
            elif poll == KVPoll.Bootstrapping:
                if not req.is_retracted and self.can_prefill_before_bootstrap(req):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.disagg.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
                    req.arrival_processed_tokens = (
                        self.counters.processed_tokens_counter
                    )
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.disagg.prefill_attempt_count += 1
                elif not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
                req.arrival_processed_tokens = self.counters.processed_tokens_counter
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return bootstrapped_reqs

    def release_memory_occupation(self):
        for req in self.queue:
            self.kv_senders.pop(req.rid, None)
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class DisaggPrefill:
    """The PD prefill mode owner, and its three lists: the bootstrap queue,
    `prefilling` (in a launched step until its commit), and `transfer` (KV
    being sent). Bootstrapped requests wait in the core waiting list between
    the first two.
    """

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        kv: KvResources,
        capacity: KvCapacity,
        streams: Streams,
        services: SchedulerServices,
        workers: Workers,
        rust_http_port: Optional[int],
        batch_result_processor: SchedulerBatchResultProcessor,
        counters: LaunchCounters,
        request_retirer: RequestRetirer,
        core_lists: CoreLists,
    ):
        spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        transfer_backend = TransferBackend(get_disagg().disaggregation_transfer_backend)
        attn_cp_cpu_group = get_parallel().attn_cp_group.cpu_group
        attn_tp_cpu_group = get_parallel().attn_tp_group.cpu_group
        pp_group = get_parallel().pp_group
        tp_size = get_parallel().tp_size
        tp_cpu_group = get_parallel().tp_group.cpu_group
        enable_overlap = not get_schedule().disable_overlap_schedule
        token_to_kv_pool_allocator = kv.token_to_kv_pool_allocator
        draft_token_to_kv_pool = kv.draft_token_to_kv_pool
        req_to_token_pool = kv.req_to_token_pool
        tree_cache = kv.tree_cache
        max_running_requests = capacity.max_running_requests
        # Each rank owns a subset of logical KV pages, so capacity widens by N.
        max_total_num_tokens = (
            capacity.effective_max_total_num_tokens * capacity.kv_shard_widening
        )
        forward_stream = streams.forward_stream
        scheduler_stage_metrics = services.scheduler_stage_metrics
        output_streamer = services.output_streamer
        metrics_sink = services.metrics_sink
        metrics_collector = metrics_sink.metrics_collector
        kv_cache_dtype_str = workers.tp_worker.model_runner.kv_cache_dtype_str
        self.num_kv_heads = model_config.num_key_value_heads
        self.spec_algorithm = spec_algorithm
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache
        self.attn_cp_cpu_group = attn_cp_cpu_group
        self.attn_tp_cpu_group = attn_tp_cpu_group
        self.tp_size = tp_size
        self.tp_cpu_group = tp_cpu_group
        self.sliding_window_size = capacity.sliding_window_size
        self.enable_overlap = enable_overlap
        self.forward_stream = forward_stream
        self.scheduler_stage_metrics = scheduler_stage_metrics
        self.output_streamer = output_streamer
        self.metrics_sink = metrics_sink
        self.metrics_collector = metrics_collector
        self.batch_result_processor = batch_result_processor
        self.counters = counters
        self.request_retirer = request_retirer
        # Read at the commit: the chunked request the scheduler still chunks.
        self.core_lists = core_lists

        # *2 for the headroom.
        buffer_size = max_running_requests * 2
        self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
            buffer_size
        )
        self.metadata_buffers = build_metadata_buffers(
            buffer_size,
            server_args=server_args,
            model_config=model_config,
            spec_algorithm=spec_algorithm,
            custom_mem_pool=token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
        )
        self.bootstrap_queue = PrefillBootstrapQueue(
            token_to_kv_pool=token_to_kv_pool_allocator.get_kvcache(),
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            metadata_buffers=self.metadata_buffers,
            gpu_id=get_device().gpu_id,
            bootstrap_port=get_disagg().disaggregation_bootstrap_port,
            max_total_num_tokens=max_total_num_tokens,
            kv_shard_widening=capacity.kv_shard_widening,
            scheduler_stage_metrics=scheduler_stage_metrics,
            transfer_backend=transfer_backend,
            model_config=model_config,
            server_args=server_args,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            page_size=token_to_kv_pool_allocator.page_size,
            kv_cache_dtype_str=kv_cache_dtype_str,
            rust_http_port=rust_http_port,
            attn_cp_cpu_group=attn_cp_cpu_group,
            attn_tp_cpu_group=attn_tp_cpu_group,
            pp_group=pp_group,
            output_streamer=output_streamer,
            counters=counters,
            request_retirer=request_retirer,
            handle_bootstrap_failure=self.handle_bootstrap_failure,
        )
        self.kv_checksum_computer = self.bootstrap_queue.kv_checksum_computer
        self.kv_senders = self.bootstrap_queue.kv_senders
        # The requests a launched step prefills, until its commit hands
        # their KV over to the inflight queue.
        self.prefilling: List[Req] = []
        # The prefill requests that are in the middle of kv sending
        self.transfer: List[Req] = []
        # Requests with a sent chunk that are not yet on the inflight queue.
        self.pending_chunk_rids: Set[str] = set()
        # The TP-wide minimum of the result completion states is_result_ready
        # agrees on.
        self.result_completion_status = torch.empty(1, dtype=torch.int32, device="cpu")

        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

    @property
    def kv_manager(self) -> CommonKVManager:
        return self.bootstrap_queue.kv_manager

    def enqueue(self, req: Req) -> None:
        """Take a new request: it waits in the bootstrap queue for its decode peer."""
        if req.disagg is None:
            req.disagg = DisaggReqState()
        self.bootstrap_queue.add(req, self.num_kv_heads)
        req.time_stats.set_prefill_bootstrap_queue_entry_time()

    def unified_memory_move_gate(self) -> bool:
        """Whether unified-memory compaction may move pages: only when no KV
        transfer can be in flight, so it never relocates a page the transfer
        engine reads. A page is exposed from the moment its address reaches the
        peer until the transfer concludes; scheduling the final chunk clears
        `chunked_req` while earlier chunks may still be draining, and the
        request reaches the transfer list only later, in the result path."""
        return not (self.transfer or self.pending_chunk_rids)

    def lists(self) -> List[RequestList]:
        """The PD prefill lists: bootstrapping, prefilling, and sending the KV."""
        return [
            RequestList(
                name="bootstrap",
                reqs=self.bootstrap_queue.queue,
                abort=self._abort_bootstrap_reqs,
                counts_queued=True,
            ),
            RequestList(
                name="prefilling",
                reqs=self.prefilling,
                inflight=True,
                health_check=True,
                holds_kv=True,
                counts_running=True,
            ),
            # Their KV is sent until the transfer concludes.
            RequestList(
                name="transfer",
                reqs=self.transfer,
                abort=self._abort_transfer_reqs,
                holds_kv=True,
            ),
        ]

    def report_queue_stats(self, stats: SchedulerStats, priority_enabled: bool) -> None:
        """The PD prefill queue gauges: bootstrapping, and sending the KV."""
        stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
            self.bootstrap_queue.queue, priority_enabled
        )
        stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
            self.transfer, priority_enabled
        )

    def queue_log_msg(self) -> str:
        """The PD prefill part of the prefill log line."""
        return (
            f"#bootstrap-req: {len(self.bootstrap_queue.queue)}, "
            f"#inflight-req: {len(self.transfer)}, "
        )

    def load_queue_metrics(self) -> Dict[str, int]:
        """The PD prefill queue lengths of the load."""
        return dict(
            prefill_bootstrap_queue_reqs=len(self.bootstrap_queue.queue),
            prefill_inflight_queue_reqs=len(self.transfer),
        )

    def _abort_bootstrap_reqs(self, recv_req: AbortReq) -> None:
        # Abort requests that have not yet been bootstrapped
        for req in self.bootstrap_queue.queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                # The bootstrap failure the abort causes retires it.
                self.request_retirer.abort_queued_attempt(req)

                if hasattr(self.kv_senders[req.rid], "abort"):
                    self.kv_senders[req.rid].abort()
                if get_parallel().pp_size > 1:
                    prepare_abort(req, "Aborted by AbortReq.")

    def _abort_transfer_reqs(self, recv_req: AbortReq) -> None:
        # Abort in-flight requests
        for req in self.transfer:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort inflight queue request. {req.rid=}")
                if hasattr(self.kv_senders[req.rid], "abort"):
                    self.kv_senders[req.rid].abort()

    def release_memory_occupation(self) -> None:
        self.bootstrap_queue.release_memory_occupation()

    def resume_memory_occupation(self) -> None:
        self.bootstrap_queue.resume_memory_occupation()

    def intake(self) -> List[Req]:
        """The requests whose bootstrap finished; they join the core waiting list."""
        return self.bootstrap_queue.pop_bootstrapped()

    def before_forward(self, batch: ScheduleBatch) -> None:
        """Launch work ahead of the step's forward: early-send cached prefix KV,
        overlapping the suffix forward."""
        if self.enable_staging:
            # Decode allocates its staging while this forward runs.
            self.maybe_prefetch_staging_for_batch(batch)
        for req in batch.reqs:
            self.maybe_send_cached_prefix_chunk(req)

    def on_launch(self, reqs: List[Req]) -> None:
        """The launched requests prefill until their commit hands the KV over."""
        self.prefilling.extend(reqs)

    def release_aborted_chunked_req(self, req: Req) -> None:
        """Release the PD state of a chunked request whose abort is applied."""
        self.clear_pending_chunk_send(req)
        self.kv_senders[req.rid].abort()
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.disagg.pending_bootstrap = False
        self.kv_senders.pop(req.rid, None)

    def maybe_prefetch_staging_for_batch(self, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def checkpoint(self, req: Req) -> None:
        cache = self.tree_cache
        if req.disagg.pending_bootstrap and _uses_write_through_cache(cache):
            cache.advance_unpublished_req(req)
            req.prefix_len = req.extend_end
            return

        checkpoint_kv_cache(req, cache)

    def release_aborted_waiting_req(self, req: Req) -> None:
        """Tear down the PD transfer state of a waiting request the abort
        removes; the retirer releases the rest."""
        self.clear_pending_chunk_send(req)
        sender = self.kv_senders.get(req.rid)
        if sender is not None and hasattr(sender, "abort"):
            sender.abort()
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.disagg.pending_bootstrap = False
        self.kv_senders.pop(req.rid, None)

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def resolve_waiting_bootstrap(self, waiting: List[Req]) -> List[Req]:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a prefill step: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check. Returns the waiting requests left.
        """
        candidates = [req for req in waiting if not is_aborted(req)]
        if not candidates:
            return waiting
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [self.kv_senders[req.rid] for req in candidates],
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
                and req.disagg.pending_bootstrap
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                self.bootstrap_queue.finalize_bootstrap(req)
        if failed:
            waiting = [req for req in waiting if req not in failed]
        return waiting

    def has_bootstrapped_waiting_req(self, waiting: List[Req]) -> bool:
        return any(
            not req.disagg.pending_bootstrap and not is_aborted(req) for req in waiting
        )

    def is_result_ready(self, batch_result: GenerationBatchResult) -> bool:
        """Whether a launched step's result copy finished on every TP rank:
        the commit readiness gate of the overlap flat driver."""
        # 0: sampling/copy has not been submitted, so polling cannot make progress.
        state = 0
        if (
            batch_result.copy_done is not None
            and batch_result.delay_sample_func is None
        ):
            # 1: copy unfinished; 2: copy finished locally.
            state = 2 if batch_result.copy_done.query() else 1
        if self.tp_size > 1:
            status = self.result_completion_status
            status.fill_(state)
            torch.distributed.all_reduce(
                status, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
            )
            state = status.item()
        if state == 0:
            # Every rank must submit delayed sampling before entering this wait.
            raise RuntimeError("Prefill result copy must be submitted before polling")
        return state == 2

    def commit_extend(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> List[Req]:
        """The host commit of a prefill step: the requests whose prefill
        concluded hand their KV over to the transfer list. Returns the
        optimistic requests that yield back to the front of the waiting
        list."""
        return self.batch_result_processor.commit_extend(
            batch, result, _HandoffOutcome(self, result)
        )

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def poll_transfer_rids(self) -> List[str]:
        """The ids of the transfer requests whose KV transfer ended on this
        rank (Success or Failed); under PP the ranks agree on them before
        after_commit releases them."""
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [self.kv_senders[req.rid] for req in self.transfer],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return [
            req.rid
            for req, poll in zip(self.transfer, polls)
            if poll in [KVPoll.Success, KVPoll.Failed]
        ]

    def after_commit(
        self, waiting: List[Req], rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, release them.
        Returns the waiting requests: a parked optimistic request may yield to
        their front.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.transfer) == 0:
            return waiting

        done_reqs = []

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [self.kv_senders[req.rid] for req in self.transfer],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in the transfer list. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.transfer, polls):
            if rids_to_check is not None:
                if req.rid not in rids_to_check:
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
                        f"PP rank {get_parallel().pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone",
                    )
                    undone_reqs.append(req)
                    continue

            if req.disagg.pending_bootstrap:
                # Parked: prefill finished before bootstrap completed.
                if self.handle_pending_bootstrap(req, poll):
                    self.send_kv_chunk(req, last_chunk=True)
                    undone_reqs.append(req)
                elif (
                    poll == KVPoll.Bootstrapping
                    and get_disagg().disaggregation_decode_allocation_policy
                    != "prefill_complete"
                    and self.req_to_token_pool.available_size() == 0
                    and (
                        waiting
                        or (
                            self.bootstrap_queue.queue
                            and self.req_to_metadata_buffer_idx_allocator.available_size()
                            == 0
                        )
                    )
                ):
                    # Decode may have admitted different requests. Yield one
                    # slot so they can prefill, including peers still waiting
                    # for a metadata buffer in the bootstrap queue.
                    waiting = self.optimistic_release_and_requeue(req, waiting)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # todo: set Transferring correctly in backend
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                if req.return_routed_experts:
                    self.batch_result_processor._maybe_collect_routed_experts(req)
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    req.finished_reason = FINISH_LENGTH(length=0)
                # unlock the tree
                self.request_retirer.retire_finished(req, checkpoint=True)
                # FIXME: clean up req's data in transfer engine
                self.kv_senders[req.rid].clear()
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
            kv_mgr = getattr(self.kv_senders[req.rid], "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                self.kv_senders[req.rid].get_transfer_metric()
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_sink.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_sink.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.output_streamer.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )
            self.kv_senders.pop(req.rid, None)

        self.transfer = undone_reqs

        return waiting

    def handle_inflight_transfer_failure(self, req: Req) -> Optional[Exception]:
        """Conclude an inflight request whose KV transfer failed."""
        error_message = (
            f"Prefill transfer failed for request rank={get_parallel().tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        exc: Optional[Exception] = None
        try:
            self.kv_senders[req.rid].failure_exception()
        except Exception as e:
            exc = e
            error_message += f" with exception {e}"
        # Mute error message for propagated exceptions to avoid duplicate logging
        if getattr(exc, "is_from_another_rank", False):
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if req.finished_reason is None:
            req.finished_reason = FINISH_LENGTH(length=0)
        # unlock the tree
        self.request_retirer.retire_transfer_failed(req)
        if not isinstance(req.finished_reason, FINISH_ABORT):
            prepare_abort(
                req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        if self.metrics_sink.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()
        return exc

    def clear_pending_chunk_send(self, req: Req) -> None:
        """Drop `req` from the sent-but-unconcluded chunk set.

        Every path that retires a request without a `last_chunk=True` send must
        call this: a stale entry holds the unified-memory compaction gate closed
        for the process lifetime.
        """
        self.pending_chunk_rids.discard(req.rid)

    def _retire_aborted_prefill_result(self, req: Req) -> bool:
        """Release an aborted request when its last prefill result is safe."""
        self.clear_pending_chunk_send(req)
        owns_resources = (
            req.kv.holds_kv
            or req.kv.holds_mamba
            or req.disagg.metadata_buffer_index >= 0
        )
        if not owns_resources:
            # A bootstrap failure or earlier abort already retired it.
            return False

        sender = self.kv_senders.get(req.rid)
        if sender is not None:
            try:
                sender.abort()
            except Exception:
                # Transport notification is best effort; local ownership must
                # still be released or the next idle invariant check will fail.
                logger.exception("Failed to notify KV sender of abort for %s", req.rid)

        if req.to_finish is not None and not req.finished():
            req.update_finish_state()
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.disagg.pending_bootstrap = False
        self.request_retirer.retire_aborted(req)
        self.kv_senders.pop(req.rid, None)
        return True

    def handle_bootstrap_failure(self, req: Req) -> None:
        self.clear_pending_chunk_send(req)
        error_message = (
            f"Prefill bootstrap failed for request rank={get_parallel().tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            self.kv_senders[req.rid].failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # Mute error message for propagated exceptions to avoid duplicate logging
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.disagg.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_sink.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        self.request_retirer.retire_aborted(req)
        self.kv_senders.pop(req.rid, None)

    def handle_pending_bootstrap(self, req: Req, poll: KVPoll) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            return False
        elif poll == KVPoll.WaitingForInput:
            if should_force_retry(req):  # test hook
                return False
            # Metadata buffer was allocated in pop_bootstrapped before
            # the request entered the waiting queue, so finalize should not fail.
            assert self.bootstrap_queue.finalize_bootstrap(req)
            self.checkpoint(req)
            return True
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.disagg.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [self.kv_senders[req.rid]],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(req, polls[0])

    def process_prefill_chunk(
        self, chunked_req: Optional[Req], waiting: List[Req], pending: PendingSteps
    ) -> Tuple[Optional[Req], List[Req]]:
        """Advance the chunked request's bootstrap and KV send; an optimistic
        one yields to bootstrapped waiting requests. Returns the chunked
        request and the waiting requests."""
        if (req := chunked_req) is None:
            return chunked_req, waiting
        if not self.check_bootstrap(req):
            if is_aborted(req):
                # bootstrap failed
                chunked_req = None
            elif (
                get_disagg().disaggregation_decode_allocation_policy
                != "prefill_complete"
                and self.has_bootstrapped_waiting_req(waiting)
            ):
                # optimistic request yields to waiting requests
                chunked_req = None
                if any(record.step.chunked_req is req for record in pending):
                    # Its launched chunks still run; their commit requeues it.
                    self.prefilling.append(req)
                else:
                    waiting = self.optimistic_release_and_requeue(req, waiting)
            # else: still bootstrapping, keep computing without sending
        elif not self.enable_overlap:
            # With overlap, the commit of the chunk sends it, once its KV is ready.
            if get_parallel().pp_size > 1:
                # A PP ring slot sends the chunk the previous slot launched,
                # before its commit: the transfer waits for the forwards
                # launched so far, that chunk's last among them.
                self.kv_senders[
                    req.rid
                ]._early_send_wait_event = self.forward_stream.record_event()
            self.send_kv_chunk(req)
        return chunked_req, waiting

    def maybe_send_cached_prefix_chunk(self, req: Req) -> None:
        if not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get():
            return

        # Staging sends into positional grid slots, so the early-send boundary
        # must stay stable across the request's batches: snapshot the at-rest
        # prefix on the first batch. Non-staging reads the live prefix.
        if self.enable_staging and req.disagg.early_send_prefix_end is None:
            req.disagg.early_send_prefix_end = max(
                0, req.prefix_len - req.host_hit_length
            )

        if req.disagg.pending_bootstrap:
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = (
            req.disagg.early_send_prefix_end
            if self.enable_staging
            else req.prefix_len - req.host_hit_length
        )
        if cached_end <= req.disagg.start_send_idx:
            return
        if cached_end % self.token_to_kv_pool_allocator.page_size != 0:
            # DCP radix hits can end on a logical cache-page boundary that is
            # not a complete physical DCP page. The regular final send covers
            # the full range; only skip this optional early-send optimization.
            return
        # Early-send issues the KV read before this step's forward is enqueued,
        # but under overlap scheduling the PRIOR step's prefill forward may still
        # be writing these prefix pages on forward_stream. Record a completion
        # event now so the transfer worker can wait on those writes before the
        # RDMA read, instead of racing them.
        if self.enable_overlap:
            ev = torch.cuda.Event()
            ev.record(self.forward_stream)
            self.kv_senders[req.rid]._early_send_wait_event = ev
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)

    def send_kv_chunk(
        self,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        computer: Optional[KvChecksumComputer] = self.kv_checksum_computer
        if last_chunk and computer is not None:
            if is_health_check_req(req):
                digest = signature = 0
            else:
                if end_idx is None:
                    end_idx = min(req.extend_end, len(req.origin_input_ids))
                # Only the pages this prefill sends; decode digests the same
                # range of its row, past the prefix it promised at bootstrap.
                start_idx = req.disagg.decode_prefix_len
                page_indices_gpu = page_indices_for_request(
                    self, req, end_idx, start_idx=start_idx
                )
                state_indices = state_indices_for_request(
                    self, req, end_idx, start_idx=start_idx
                )
                digest = computer.compute(page_indices_gpu, state_indices)
                signature = self.bootstrap_queue.kv_checksum_signature
            self.metadata_buffers.set_kv_checksum(req, digest, signature)
        self._send_kv_chunk(req, last_chunk=last_chunk, end_idx=end_idx)

    def _send_kv_chunk(
        self,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.disagg.start_send_idx
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx if end_idx is not None else min(req.extend_end, transfer_input_len)
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size
            if self.enable_staging:
                # Staging identifies chunks positionally against a uniform
                # prefetched grid, so non-last sends must end on a grid
                # boundary; the remainder rides with the next send.
                grid_tokens = staging_grid_tokens(
                    get_schedule().chunked_prefill_size, page_size
                )
                base = req.disagg.decode_prefix_len
                end_idx = base + ((end_idx - base) // grid_tokens) * grid_tokens

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        state_indices: Optional[List] = None
        if last_chunk:
            self.metadata_buffers.set_buf(req)

            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # with which decode registered the destination row.
            seq_len = min(req.extend_end, transfer_input_len)
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.translate_mamba_indices(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            req.kv.req_pool_idx
                        ]
                    )
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                kvcache = self.token_to_kv_pool_allocator.get_kvcache()
                request_window = kvcache.request_window
                # Under encoder replay the scheduler is not hybrid SWA and keeps no
                # window size; the pool holds it.
                window_size = (
                    self.sliding_window_size
                    if request_window is None
                    else kvcache.sliding_window
                )
                window_start = max(req.disagg.decode_prefix_len, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                if request_window is not None:
                    return request_window.page_indices(
                        int(req.kv.req_pool_idx), window_start, seq_len
                    )
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_swa_indices_for_transfer(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _full_kv_pages_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _minimax_payload():
                return minimax_index_transfer_pages(
                    self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :seq_len],
                    page_size=page_size,
                    prefix_len=req.disagg_decode_prefix_len,
                )

            def _dsa_pages_payload():
                # Decode already holds the indexer rows of the prefix it cached
                # (decode_prefix_len, page-aligned), and its rows for a prefix
                # still being restored point at no page yet: send only the
                # suffix, positionally matching decode's _dsa_pages_payload.
                return suffix_page_indices(
                    self.req_to_token_pool.req_to_token,
                    req.kv.req_pool_idx,
                    req.disagg.decode_prefix_len,
                    seq_len,
                    page_size,
                )

            def _dsa_tail_payload():
                return get_dsa_tail_state_indices(
                    self.token_to_kv_pool_allocator.get_kvcache(),
                    req.kv.req_pool_idx,
                    seq_len,
                )

            def _qsa_pending_payload():
                # Raw index-K/RoPE state is one full compression-group ring per
                # request, addressed by the request-pool slot rather than KV pages.
                return get_qsa_pending_state_indices(req)

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.kv.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _request_state_payload():
                kvcache = self.token_to_kv_pool_allocator.get_kvcache()
                return kvcache.request_state_transfer_indices(
                    int(req.kv.req_pool_idx), c128_seq_len
                )

            def _request_row_payload():
                return np.array([req.kv.req_pool_idx], dtype=np.int32)

            state_types = self.bootstrap_queue.kv_manager.kv_args.state_types
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.QSA_PENDING: _qsa_pending_payload,
                StateType.QSA_COMPRESSED: _full_kv_pages_payload,
                StateType.SWA: _swa_payload,
                StateType.DSA: _dsa_pages_payload,
                StateType.DSA_TAIL: _dsa_tail_payload,
                StateType.MINIMAX_INDEX_K: _minimax_payload,
                StateType.MINIMAX_DENSE_KV: _full_kv_pages_payload,
                StateType.SWA_RING: _swa_ring_payload,
                StateType.DSV4_REQUEST_STATE: _request_state_payload,
                StateType.REQUEST_ROW: _request_row_payload,
                StateType.BLOCK_SCALE: _full_kv_pages_payload,
                StateType.BLOCK_SCALE_SWA: _swa_payload,
            }
            if _is_npu and isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(),
                DeepSeekV4TokenToKVPool,
            ):
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )

                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        req.kv.req_pool_idx,
                        seq_len,
                        page_size,
                        prefix_len=req.disagg.decode_prefix_len,
                    )
                )
            state_indices = [
                payloads[st]() if st in payloads else None for st in state_types
            ]

        transfer_chunk_tokens = self.kv_senders[req.rid].get_max_transfer_tokens()
        if self.enable_staging:
            # One sender.send per grid slot; the sender's cumulative page
            # counter marks only the final sub-send of the final chunk as
            # is_last, routing aux/state correctly.
            transfer_chunk_tokens = staging_grid_tokens(
                get_schedule().chunked_prefill_size, page_size
            )
        if transfer_chunk_tokens is not None:
            # Prefill cache hits can leave more KV to transfer than the DCP pack buffer holds.
            segments = compute_grid_segments(
                start_idx,
                end_idx,
                req.disagg.decode_prefix_len,
                transfer_chunk_tokens,
            )
        else:
            segments = [(start_idx, end_idx)]

        for seg_start, seg_end in segments:
            is_final_segment = seg_end == end_idx
            raw_kv_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, seg_start:seg_end
            ]
            # Unified memory: req_to_token holds VIRTUAL ids; the transfer needs
            # physical ones. Per segment, since each is its own gather.
            kv_indices = (
                self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                    raw_kv_indices
                )
            )
            sender = self.kv_senders[req.rid]
            indices_event = None
            if getattr(sender, "supports_host_tensor_indices", False) and not (
                sender.kv_mgr.enable_all_cp_ranks_for_transfer
                and not get_parallel().enable_dsa_cache_layer_split
            ):
                # The schedule stream may wait on the in-flight forward; move the
                # D2H wait to the transfer worker so scheduling can continue.
                page_indices_dev = kv_indices[::page_size] // page_size
                page_indices = torch.empty(
                    page_indices_dev.shape,
                    dtype=page_indices_dev.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                page_indices.copy_(page_indices_dev, non_blocking=True)
                indices_event = torch.cuda.Event()
                indices_event.record(torch.cuda.current_stream())
            else:
                page_indices = kv_to_page_indices(kv_indices, page_size)
            segment_is_last = last_chunk and is_final_segment
            if not sender.should_send_kv_chunk(len(page_indices), segment_is_last):
                continue
            send_state_indices = state_indices if segment_is_last else None
            if indices_event is not None:
                sender.send(
                    page_indices,
                    send_state_indices,
                    num_kv_tokens=seg_end - seg_start,
                    indices_event=indices_event,
                )
            else:
                sender.send(
                    page_indices,
                    send_state_indices,
                    num_kv_tokens=seg_end - seg_start,
                )
        req.disagg.start_send_idx = end_idx
        # A last chunk needs no entry: every `last_chunk=True` call site has
        # already put the request on the transfer list.
        if last_chunk:
            self.pending_chunk_rids.discard(req.rid)
        else:
            self.pending_chunk_rids.add(req.rid)

    def optimistic_release_and_requeue(self, req: Req, waiting: List[Req]) -> List[Req]:
        """Release KV cache and requeue an optimistic prefill request: to the
        front of the waiting requests, which it returns, or to the bootstrap
        queue."""
        max_attempts = get_disagg().optimistic_prefill_attempts
        uses_write_through_cache = _uses_write_through_cache(self.tree_cache)
        if not uses_write_through_cache:
            checkpoint_kv_cache(req, self.tree_cache)
        # The cached prefix is evictable once the KV is released. Its length
        # (capped at what a retry can match) seeds the retry's storage baseline,
        # so an evicted prefix is looked up in L3 once before it is recomputed.
        yielded_prefix_len = (
            0
            if req.skip_radix_cache_insert
            else min(
                req.kv.cache_protected_len,
                req._compute_max_prefix_len(len(req.full_untruncated_fill_ids)),
            )
        )
        # The checkpoint above already handed the prefill KV to the tree; the
        # request is not finished, so the release only frees the rest.
        self.request_retirer.requeue_attempt(req)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.disagg.start_send_idx = 0
        self.clear_pending_chunk_send(req)  # re-sends from scratch
        req.disagg.decode_prefix_len = 0
        req.disagg.early_send_prefix_end = None
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
        req.disagg.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        req.advance_cache_request_handle()
        # A fresh lookup budget for the new attempt, as after a retraction.
        req.storage_prefetch_retry_attempts = 0
        req.storage_prefetch_last_match_len = yielded_prefix_len or None
        if (
            req.disagg.prefill_attempt_count >= max_attempts
            and get_disagg().disaggregation_decode_allocation_policy
            != "prefill_complete"
        ):
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill attempts "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )
            self.bootstrap_queue.queue.append(req)
        else:
            req.disagg.prefill_attempt_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill yielded "
                f"({req.disagg.prefill_attempt_count}/{max_attempts} attempts used)"
            )
            if self.metrics_sink.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            req.arrival_processed_tokens = self.counters.processed_tokens_counter
            waiting = [req, *waiting]
        return waiting


def _abort_before_handoff(req: Req, error: str) -> None:
    """Abort a concluded PD prefill request now: the commit retires it
    instead of handing its KV over."""
    prepare_abort(req, error, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


class _HandoffOutcome:
    """PD prefill: a concluded row hands its KV over to the transfer list
    (or aborts, or yields an optimistic request back to waiting); the
    chunked request's row sends its chunk under overlap. Only aborted
    requests stream."""

    def __init__(self, owner: DisaggPrefill, result: GenerationBatchResult):
        self.owner = owner
        self.processor = owner.batch_result_processor
        # The chunked request the scheduler still chunks, at this commit.
        self.chunked_req = owner.core_lists.chunked_req
        # The draft state the decode node's first verify starts from.
        self.draft_input = (
            result.next_draft_input if owner.spec_algorithm.is_eagle() else None
        )
        self._draft_cpu = None
        # Requests whose last launched chunk this step ran: they leave
        # prefilling, to the inflight queue, back to waiting, or finished.
        self.concluded: List[Req] = []
        self.aborted: List[Req] = []
        self.requeued: List[Req] = []

    def _draft_tensors_cpu(self):
        if self._draft_cpu is None:
            draft_input = self.draft_input
            self._draft_cpu = (
                draft_input.hidden_states.to("cpu", non_blocking=False),
                (
                    draft_input.dsa_topk_indices.to("cpu", non_blocking=False)
                    if draft_input.dsa_topk_indices is not None
                    else None
                ),
            )
        return self._draft_cpu

    def _skip_logprobs(self, commit, i: int, req: Req) -> None:
        if commit.batch.return_logprob:
            self.processor.apply_prefill_logprobs(commit, i, req, store=False)

    def _drop(self, commit, i: int, req: Req) -> None:
        if self.owner._retire_aborted_prefill_result(req):
            req.time_stats.set_completion_time()
            self.aborted.append(req)
        self._skip_logprobs(commit, i, req)

    def conclude(self, commit, i: int, req: Req, next_token_id: int) -> None:
        owner = self.owner
        processor = self.processor
        self.concluded.append(req)
        req.time_stats.set_prefill_finished_time()

        # Its KV never landed: abort instead of sending it to decode.
        if req.discard_output_reason is not None:
            req.to_finish = req.discard_output_reason
        if is_aborted(req):
            self._drop(commit, i, req)
            return

        # Test hook: exercise the release/requeue retry path.
        if req.disagg.pending_bootstrap and should_force_retry(req):
            self.requeued = owner.optimistic_release_and_requeue(req, self.requeued)
            self._skip_logprobs(commit, i, req)
            return

        if req.return_sampling_mask:
            sampling_mask_finish_reason = processor.sampling_mask_finish_reason(
                commit, i
            )
            if sampling_mask_finish_reason is not None:
                req.to_finish = sampling_mask_finish_reason
                req.time_stats.trace_ctx.abort(
                    abort_info={"reason": sampling_mask_finish_reason.message}
                )
                self._drop(commit, i, req)
                return

        req.output_ids.append(next_token_id)
        if req.grammar is not None:
            processor.apply_prefill_grammar(
                commit, req, next_token_id, abort=_abort_before_handoff
            )
            if is_aborted(req):
                self._drop(commit, i, req)
                return

        owner.checkpoint(req)
        owner.transfer.append(req)
        draft_input = self.draft_input
        if draft_input is not None:
            draft_hidden_states_cpu, draft_dsa_topk_indices_cpu = (
                self._draft_tensors_cpu()
            )
            req.output_topk_p = draft_input.topk_p[i]
            req.output_topk_index = draft_input.topk_index[i]
            req.hidden_states_tensor = draft_hidden_states_cpu[i].clone()
            if draft_dsa_topk_indices_cpu is not None:
                req.output_dsa_topk_indices = draft_dsa_topk_indices_cpu[i].clone()
            else:
                req.output_dsa_topk_indices = None
        else:
            req.hidden_states_tensor = None
            req.output_dsa_topk_indices = None
        if commit.batch.return_logprob:
            processor.apply_prefill_logprobs(commit, i, req)
        if req.return_sampling_mask:
            processor.add_sampling_mask_return_values(i, req, commit.logits_output)
        if get_disagg().disaggregation_decode_allocation_policy == "prefill_complete":
            owner.kv_senders[req.rid].mark_prefill_complete()
        if not req.disagg.pending_bootstrap:
            owner.send_kv_chunk(req, last_chunk=True)
        req.time_stats.set_prefill_transfer_queue_entry_time()

    def continue_chunk(self, commit, i: int, req: Req) -> None:
        owner = self.owner
        # Still chunking iff its next chunk was launched: either it is
        # still the chunked request, or its final chunk (extend_end
        # reaching the end of the input) is in flight. A yielded req
        # is neither, so do its deferred release here.
        still_chunking = self.chunked_req is req or (
            req.extend_end is not None and req.extend_end >= len(req.origin_input_ids)
        )
        if not still_chunking:
            self.concluded.append(req)
        # Abort is terminal. Do not requeue an aborted optimistic
        # request merely because bootstrap is still pending.
        if is_aborted(req):
            if not still_chunking and owner._retire_aborted_prefill_result(req):
                req.time_stats.set_completion_time()
                self.aborted.append(req)
            self._skip_chunk_logprobs(commit, i, req)
            req.time_stats.set_last_chunked_prefill_finish_time()
            return

        if req.disagg.pending_bootstrap and not still_chunking:
            self.requeued = owner.optimistic_release_and_requeue(req, self.requeued)
            self._skip_chunk_logprobs(commit, i, req)
            req.time_stats.set_last_chunked_prefill_finish_time()
            return

        if commit.batch.return_logprob:
            self.processor.apply_chunked_prefill_logprobs(commit, i, req)

        # In non-overlap-mode, KV is sent in process_prefill_chunk
        # Only send when req's sender is initialized
        if owner.enable_overlap and not req.disagg.pending_bootstrap:
            assert req.disagg.metadata_buffer_index >= 0, (
                f"Req {req.rid} does not have metadata buffer allocated"
            )
            chunk_end = commit.batch.prefix_lens[i] + commit.batch.extend_lens[i]
            owner.send_kv_chunk(
                req,
                last_chunk=False,
                end_idx=min(chunk_end, len(req.origin_input_ids)),
            )
        req.time_stats.set_last_chunked_prefill_finish_time()

    def _skip_chunk_logprobs(self, commit, i: int, req: Req) -> None:
        if commit.batch.return_logprob:
            self.processor.apply_chunked_prefill_logprobs(commit, i, req, store=False)

    def finish(self, commit) -> List[Req]:
        owner = self.owner
        if self.concluded:
            owner.prefilling = [r for r in owner.prefilling if r not in self.concluded]
        if self.aborted:
            owner.output_streamer.stream_output(
                self.aborted,
                any(req.return_logprob for req in self.aborted),
            )
        return self.requeued
