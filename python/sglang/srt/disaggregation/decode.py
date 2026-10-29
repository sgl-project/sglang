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

import contextlib
import hashlib
import logging
import random
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import KVTransferDestination, StateType
from sglang.srt.disaggregation.checksum import (
    KvChecksumComputer,
    is_health_check_req,
    page_indices_for_request,
    state_indices_for_request,
)
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.minimax import minimax_index_transfer_pages
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    _is_fake_transfer,
    all_reduce_flags,
    build_kv_layer_ids,
    build_staging_slot_metadata,
    get_dsa_tail_state_indices,
    get_kv_class,
    get_kv_transfer_buf_infos,
    get_pd_kv_layout,
    get_qsa_pending_state_indices,
    is_mla_backend,
    is_unadmitted_reject,
    kv_layout_signature,
    poll_and_all_reduce,
    poll_and_all_reduce_pp,
    poll_and_all_reduce_with_staging,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    kv_private_swa_tokens,
    kv_private_tokens,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocation import ensure_mamba_capacity
from sglang.srt.mem_cache.allocation_sizing import get_mamba_tracking_slots
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    TreeLock,
)
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    discard_kv_cache_backup,
    dsv4_needs_rebootstrap,
    finish_restore_kv_cache,
    kv_to_page_indices,
    match_kv_cache,
    page_align_floor,
    release_kv_cache,
    restore_kv_cache,
    start_restore_kv_cache,
    suffix_page_indices,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv_cache_builder import decode_retraction_max_tokens
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KVCache,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage, kv_storage_of
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_QUEUE,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_memory,
    get_parallel,
)
from sglang.srt.utils import ceil_align, get_num_new_pages, is_npu
from sglang.srt.utils.common import async_h2d
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.kv.storage import KVStorage

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


def _bind_root_prefix(req: Req, tree_cache: BasePrefixCache) -> None:
    """Start a decode-radix request that owns its whole KV row at the root."""
    req.prefix_len = 0
    req.last_node = tree_cache.root_node_handle(req.extra_key)
    req.last_host_node = req.last_node
    req.best_match_node = req.last_node
    req.lock = None
    req.kv.cache_protected_len = 0
    req.kv.cache_inserted_len = 0
    req.num_matched_prefix_tokens = 0
    req.host_hit_length = 0


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    # Mirrors ReqToTokenPool.register_on_alloc_rows.
    _on_alloc_rows: Optional[Callable[[List[int]], None]] = None
    # No recurrent state (`ReqToTokenPool.mamba_pool`, `ReqToTokenPool.state`).
    holds_state_slots = False
    state = None
    mamba_pool = None
    mamba_allocator = None
    mamba_ckpt_pool = None

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
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(1, self._alloc_size))
        # Slot-reuse generation counter; mirrors ReqToTokenPool. Required even
        # here: HybridMambaDecodeReqToTokenPool borrows this __init__ while
        # inheriting ReqToTokenPool.alloc, which bumps it.
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
        self._aux_cache: Any = None

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def reset_aux_cache_allocator(self) -> None:
        pass

    def schedulable_token_capacity(self, physical_capacity: int) -> int:
        return physical_capacity

    def alloc_aux_to_lengths(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        target_seq_lens_cpu: torch.Tensor,
    ) -> None:
        pass

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.kv.holds_kv]
        assert all(reqs[i].kv.kv_allocated_len > 0 for i in reusing), (
            "a reused row must carry allocated KV"
        )

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        if self._on_alloc_rows is not None and select_index:
            self._on_alloc_rows(select_index)
        offset = 0
        for r in reqs:
            if not r.kv.holds_kv:
                r.kv.req_pool_idx = select_index[offset]
                self.req_generation[r.kv.req_pool_idx] += 1
                offset += 1
        return [r.kv.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.kv.holds_kv, "request must have req_pool_idx"
        self.free_slots.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        # req_generation must stay monotonic; see ReqToTokenPool.clear().

    def register_on_alloc_rows(self, hook: Callable[[List[int]], None]) -> None:
        assert self._on_alloc_rows is None
        self._on_alloc_rows = hook


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: Mamba2CacheParams,
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
        speculative_eagle_topk: Optional[int] = None,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape: Optional[Tuple[int, int]] = None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = get_mamba_tracking_slots(
            extra_buffer=True, overlap=enable_overlap_schedule
        )
        self.mamba_initial_tracking_slots = get_mamba_tracking_slots(
            extra_buffer=enable_mamba_extra_buffer,
            overlap=enable_overlap_schedule,
            lazy=False,
        )
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        slots_per_req = 1 + self.mamba_initial_tracking_slots
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self._init_mamba_pool(
            mamba_size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            mamba_envelope_layout=mamba_envelope_layout,
            enable_linear_replayssm_spec=enable_linear_replayssm_spec,
            short_conv_layer_ids=short_conv_layer_ids,
            short_conv_state_shape=short_conv_state_shape,
            ngram_context_len=ngram_context_len,
            ngram_eos_token_id=ngram_eos_token_id,
        )

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.state.slots.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    is_rebootstrap: bool = False
    host_staged: bool = False

    # HiCache Status
    prefix_match: Optional[DecodePrefixMatch] = None
    hicache_restored_kv_indices: Optional[torch.Tensor] = None
    # The lock held on the restored node until the commit hands it to the req.
    hicache_restore_lock: Optional[TreeLock] = None
    hicache_load_consumer_index: int = -1
    hicache_restore_status: HiCacheRestoreResult = HiCacheRestoreResult.PENDING

    @property
    def seqlen(self) -> int:
        return self.req.seqlen

    @property
    def priority(self) -> Optional[int]:
        return self.req.priority


class DecodePreallocQueue(DecodeHiCachePreallocMixin):
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        parallel = get_parallel()
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache
        self.host_pool = None
        self.host_reserved_tokens = 0
        self.gloo_group = parallel.attn_tp_group.cpu_group
        # Destinations visible to prefill but not yet on the transfer queue.
        self._num_published_destinations = 0
        self.tp_rank = parallel.tp_rank
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = parallel.pp_rank
        self.pp_size = parallel.pp_size
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        # In-flight authoritative room -> DP-rank lookups, consumed below.
        self._prefill_dp_rank_queries: Dict[
            str, Tuple[Tuple[int, ...], Future[Dict[str, int]]]
        ] = {}
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        # Retracted requests staged for rebootstrap while generation is paused.
        # Enqueued into ``self.queue`` only on ``continue_generation`` so the
        # prefix KV is recomputed under the post-retract (updated) weights.
        # NOTE: requests held here are not reachable by ``/abort_request``; to
        # support aborting them we would need an additional fix in the
        # scheduler. In practice this shouldn't arise in the RL scenario.
        self.held_rebootstrap_reqs: List[Req] = []
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)

        if (
            self.scheduler.tp_worker.is_hybrid_swa
            and not self._uses_swa_tail_prealloc()
        ):
            # Fallback for SWA allocators that still allocate the SWA pool at
            # full prompt length.
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

        enable_kv_checksum = get_disagg().disaggregation_enable_kv_checksum
        self.kv_checksum_signature = 0
        # Test-only mismatch injector. Its own generator, seeded per rank: the
        # global `random` is seeded identically on every TP rank, so drawing
        # from it would inject the same requests everywhere and never exercise
        # the mismatch all-reduce, while skipping draws on some ranks would
        # desync that shared stream.
        self.kv_checksum_mismatch_injector = random.Random(
            hash((self.scheduler.random_seed, get_parallel().attn_tp_rank))
        )
        if enable_kv_checksum:
            kv_args = self.kv_manager.kv_args
            self.kv_checksum_signature = kv_layout_signature(kv_args)
            self.scheduler.kv_checksum_computer = KvChecksumComputer(
                device=torch.device(f"cuda:{get_device().gpu_id}"),
                kv_data_ptrs=kv_args.kv_data_ptrs,
                kv_item_lens=kv_args.kv_item_lens,
                state_data_ptrs=kv_args.state_data_ptrs,
                state_item_lens=kv_args.state_item_lens,
            )
        else:
            self.scheduler.kv_checksum_computer = None

    def _uses_swa_tail_prealloc(self) -> bool:
        return (
            (
                self.token_to_kv_pool.layer_router.sub_pool(IdSpaceKind.SLIDING_WINDOW)
                is not None
                or isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
            )
            and self.token_to_kv_pool_allocator.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
        )

    def _uses_swa_reservation(self) -> bool:
        return (
            self._uses_swa_tail_prealloc()
            or self.token_to_kv_pool_allocator.prealloc_fits_assumes_reclaim()
        )

    def _required_admission_tokens(
        self, req: Req, allocated_tokens: int, prefix_len: int, retractable_tokens: int
    ) -> int:
        return max(
            allocated_tokens + self.num_reserved_decode_tokens,
            self._rebootstrap_prefill_len(req)
            - prefix_len
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
            - retractable_tokens,
        )

    def _prealloc_reservation_fits(
        self,
        full_tokens: int,
        swa_tokens: int,
        *,
        full_allocatable_tokens: int,
        swa_allocatable_tokens: Optional[int],
    ) -> bool:
        return self.token_to_kv_pool_allocator.prealloc_fits(
            self.tree_cache,
            full_tokens,
            swa_tokens,
            full_budget_tokens=full_allocatable_tokens,
            swa_budget_tokens=swa_allocatable_tokens,
        )

    def _reclaim_swa_tail_capacity(
        self, swa_tail_len: int, req_id: str, *, full_len: int = 0
    ) -> Optional[str]:
        allocator = self.token_to_kv_pool_allocator
        page_size = allocator.page_size
        shortfall = allocator.reclaim_for_prealloc(
            self.tree_cache,
            ceil_align(full_len, page_size),
            ceil_align(swa_tail_len, page_size),
        )
        return None if shortfall is None else f"{shortfall}, req={req_id}"

    # SWA caches expose full-attention accounting through full_* accessors.
    def _radix_full_evictable(self) -> int:
        if self.scheduler.tp_worker.is_hybrid_swa:
            return self.tree_cache.full_evictable_size()
        return self.tree_cache.evictable_size()

    def _radix_full_protected(self) -> int:
        if self.scheduler.tp_worker.is_hybrid_swa:
            return self.tree_cache.full_protected_size()
        return self.tree_cache.protected_size()

    def _radix_full_available(self) -> int:
        if self.scheduler.tp_worker.is_hybrid_swa:
            return self.token_to_kv_pool_allocator.full_available_size()
        return self.token_to_kv_pool_allocator.available_size()

    def _swa_tail_len(self, seq_len: int) -> int:
        if not self._uses_swa_tail_prealloc() or seq_len <= 0:
            return max(seq_len, 0)

        window_size = self.scheduler.sliding_window_size
        if window_size is None or window_size <= 0:
            return seq_len

        page_size = self.token_to_kv_pool_allocator.page_size
        if get_disagg().disaggregation_decode_enable_radix_cache:
            # Keep enough SWA before the page-aligned radix-cache insert
            # boundary for the cached key to contain a complete window.
            # `seq_len - 1` is the last committed position.
            window_start = max(0, seq_len - 1 - max(window_size, page_size))
        else:
            window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        return seq_len - window_start

    def _swa_retractable_len(self, req: Req) -> int:
        if not self._uses_swa_tail_prealloc():
            return len(req.origin_input_ids) + len(req.output_ids)
        return self._swa_tail_len(len(req.origin_input_ids)) + len(req.output_ids)

    def _prealloc_kv_lens(self, req: Req) -> Tuple[int, int]:
        allocated_kv_len = self._pre_alloc_fill_len(req)
        if self._uses_swa_tail_prealloc():
            return allocated_kv_len, self._swa_tail_len(allocated_kv_len)
        return allocated_kv_len, allocated_kv_len

    def _prealloc_required_tokens(self, req: Req) -> Tuple[int, int]:
        full_len, swa_len = self._prealloc_kv_lens(req)
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size > 1:
            # Match the allocator, which charges whole pages for both pools.
            full_len = ceil_align(full_len, page_size)
            swa_len = ceil_align(swa_len, page_size)
        swa_reserved = self.num_reserved_decode_tokens
        if get_memory().disable_radix_cache:
            swa_reserved = 0
        return (
            full_len + self.num_reserved_decode_tokens,
            swa_len + swa_reserved,
        )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_parallel().attn_tp_size
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = get_parallel().dp_rank
        kv_args.kv_cache_dtype_str = (
            self.scheduler.tp_worker.model_runner.kv_cache_dtype_str
        )
        kv_args.kv_layout = get_pd_kv_layout(self.token_to_kv_pool)
        transfer_kv_pool = (
            self.scheduler.hisparse_coordinator.mem_pool_host
            if self.scheduler.enable_hisparse
            else self.token_to_kv_pool
        )
        storage = kv_storage_of(
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            draft_token_to_kv_pool=self.draft_token_to_kv_pool,
            num_hidden_layers=self.scheduler.model_config.num_hidden_layers,
        )
        # HiSparse registers its host pool instead; unified memory refuses it.
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            transfer_kv_pool.get_contiguous_buf_infos()
            if self.scheduler.enable_hisparse
            else get_kv_transfer_buf_infos(storage)
        )
        kv_data_mem_kinds = (
            ["DRAM"] * len(kv_data_ptrs)
            if self.scheduler.enable_hisparse
            else ["VRAM"] * len(kv_data_ptrs)
        )
        if self.scheduler.enable_hisparse and isinstance(
            self.token_to_kv_pool, DeepSeekV4TokenToKVPool
        ):
            device_kv_data_ptrs, device_kv_data_lens, device_kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
            c4_layer_num = self.scheduler.hisparse_coordinator.mem_pool_host.layer_num
            kv_data_ptrs += device_kv_data_ptrs[c4_layer_num:]
            kv_data_lens += device_kv_data_lens[c4_layer_num:]
            kv_item_lens += device_kv_item_lens[c4_layer_num:]
            kv_data_mem_kinds += ["VRAM"] * len(device_kv_data_ptrs[c4_layer_num:])
        num_draft_entries = 0
        if self.draft_token_to_kv_pool is not None:
            # Draft KV shares target virtual ids. Unified target KV is transferred
            # with physical ids, so it needs a separate draft index vector.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            kv_data_mem_kinds += ["VRAM"] * len(draft_kv_data_ptrs)
            num_draft_entries = len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.num_draft_entries = num_draft_entries
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.token_to_kv_pool,
            draft_token_to_kv_pool=self.draft_token_to_kv_pool,
            num_draft_entries=num_draft_entries,
            num_hidden_layers=self.scheduler.model_config.num_hidden_layers,
            storage=storage,
        )
        if self.transfer_backend == TransferBackend.NIXL:
            kv_args.kv_data_mem_kinds = kv_data_mem_kinds
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        setup_state_kv_args(
            kv_args,
            storage,
            self.draft_token_to_kv_pool,
            total_kv_layers=self.scheduler.model_config.num_hidden_layers,
        )
        if get_disagg().disaggregation_decode_host_receive_threshold < 1:
            pool = self.token_to_kv_pool
            group = self.tree_cache.host_pool_group
            if (kv_args.state_types and kv_args.state_types != [StateType.SWA]) or any(
                spec.indices_from_pool not in (PoolName.KV, PoolName.SWA)
                for spec in self.tree_cache.sidecar_pool_specs
            ):
                raise ValueError(
                    "Host receive supports full KV and separate paged SWA pools"
                )
            self.host_pool = group.get_pool(PoolName.KV)
            self.host_reserved_tokens = decode_retraction_max_tokens(
                self.req_to_token_pool, pool
            )
            if self.host_pool.logical_size < self.host_reserved_tokens + pool.page_size:
                raise ValueError(
                    "Host pool must hold a retraction and a receive page; "
                    "increase --hicache-size or --hicache-ratio"
                )

            def pool_names(source):
                return {source} | {
                    spec.pool_name
                    for spec in self.tree_cache.sidecar_pool_specs
                    if spec.indices_from_pool == source
                }

            device_buffers, host_buffers = group.get_contiguous_buf_infos(
                pool_names(PoolName.KV)
            )
            if device_buffers != (
                kv_args.kv_data_ptrs,
                kv_args.kv_data_lens,
                kv_args.kv_item_lens,
            ):
                raise ValueError(
                    "Host receive buffers must match the KV transfer layout"
                )
            (
                kv_args.host_kv_data_ptrs,
                kv_args.host_kv_data_lens,
                kv_args.host_kv_item_lens,
            ) = host_buffers
            self.host_swa_pool = None
            self.host_swa_reserved_tokens = 0
            if kv_args.state_types:
                self.host_swa_pool = group.get_pool(PoolName.SWA)
                self.host_swa_reserved_tokens = min(
                    decode_retraction_max_tokens(
                        self.req_to_token_pool,
                        # The one state entry here is the window's.
                        storage.view(storage.wire_entries()[0].sub_pool),
                    ),
                    ceil_align(
                        self.scheduler.sliding_window_size + pool.page_size - 1,
                        pool.page_size,
                    ),
                )
                if self.host_swa_pool.logical_size < (
                    self.host_swa_reserved_tokens + pool.page_size
                ):
                    raise ValueError(
                        "SWA host pool must hold a retraction and a receive page; "
                        "increase --hicache-size or --hicache-ratio"
                    )
                device_state, host_state = group.get_contiguous_buf_infos(
                    pool_names(PoolName.SWA)
                )
                if device_state != (
                    kv_args.state_data_ptrs[0],
                    kv_args.state_data_lens[0],
                    kv_args.state_item_lens[0],
                ):
                    raise ValueError(
                        "Host receive buffers must match the SWA transfer layout"
                    )
                (
                    kv_args.host_state_data_ptrs,
                    kv_args.host_state_data_lens,
                    kv_args.host_state_item_lens,
                ) = ([values] for values in host_state)

        kv_args.ib_device = get_disagg().disaggregation_ib_device
        kv_args.gpu_id = get_device().gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        if (
            get_disagg().disaggregation_decode_host_receive_threshold < 1
            and not kv_manager.supports_host_destination
        ):
            raise ValueError(
                "Transfer backend does not support decode host KV destinations"
            )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool.layer_router.view(
                IdSpaceKind.FULL
            )
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = (
                    self.scheduler.model_config.get_total_num_kv_heads()
                )
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                staging_slots = build_staging_slot_metadata(
                    kv_layer_ids=kv_args.kv_layer_ids,
                    num_draft_entries=num_draft_entries,
                    kv_pool=kv_pool,
                    draft_kv_pool=self.draft_token_to_kv_pool,
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

    def add(
        self, req: Req, is_retracted: bool = False, is_rebootstrap: bool = False
    ) -> None:
        """Add a request to the pending queue.

        ``is_rebootstrap`` marks a PD true-retraction request whose prefix KV
        must be recomputed by the original prefill worker under the current
        weights (rather than resumed from stale CPU KV). It otherwise follows the
        same bootstrap-handshake path as a fresh request. Early allocation
        dispatches ``/generate`` after ``send_metadata``; deferred allocation
        dispatches once the peer is resolved, before waiting for completion.
        """
        # See `PrefillBootstrapQueue.add`. A retracted or rebootstrapping
        # request owns a host KV backup that `retracted_queue` releases, and by
        # this point carries none of the markers `is_unadmitted_reject` reads,
        # so take the caller's word for it rather than sniffing.
        if not is_retracted and not is_rebootstrap and is_unadmitted_reject(req):
            self.scheduler.retire_unadmitted_request(req)
            return
        if is_retracted and dsv4_needs_rebootstrap(self.token_to_kv_pool_allocator):
            if req.output_ids:
                req.pd_rebootstrap_forced_output_id = req.output_ids.pop()
            req.pd_rebootstrap_in_progress = True
            req.time_stats.set_retract_time()
            is_retracted = False
            is_rebootstrap = True

        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(
                req, is_rebootstrap=is_rebootstrap
            )

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                self._init_receiver(decode_req, prefill_dp_rank)
                return

            self.pending_reqs.append(decode_req)

    def _match_prefix_and_lock(self, req: Req) -> DecodePrefixMatch:
        max_prefix_len = None
        if self._uses_swa_tail_prealloc():
            fill_len = self._pre_alloc_fill_len(req)
            max_prefix_len = fill_len - self._swa_tail_len(fill_len)
        # Match and lock only reusable FULL KV. The entire SWA tail must be
        # freshly allocated, including when the prefix comes from L2/L3.
        result = match_kv_cache(
            req,
            self.tree_cache,
            req.origin_input_ids,
            cow_mamba=self.tree_cache.supports_mamba(),
            max_prefix_len=max_prefix_len,
        )
        req.lock = self.tree_cache.lock(result.last_device_node)
        return self._build_decode_prefix_match(
            req, result, max_prefix_len=max_prefix_len
        )

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(
        self, req: Req, is_rebootstrap: bool = False
    ) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE if _is_fake_transfer(req) else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(
            req=req, kv_receiver=kv_receiver, is_rebootstrap=is_rebootstrap
        )
        self.queue.append(decode_req)
        return decode_req

    def hold_rebootstrap(self, req: Req) -> None:
        """Stage a retracted request for rebootstrap without enqueuing it yet.

        Retraction is always paired with a weight update
        (``pause_generation(mode="retract")`` -> ``update_weights`` ->
        ``continue_generation``). Enqueuing the rebootstrap into ``self.queue``
        here would leave the preallocation queue non-empty, which makes the
        scheduler non-idle so ``update_weights``' post-update cache flush
        asserts and crashes the decode worker. Instead we hold the request and
        enqueue it from ``enqueue_held_rebootstrap`` on resume, so its prefix KV
        is recomputed by the prefill worker under the updated weights.
        """
        self.held_rebootstrap_reqs.append(req)

    def enqueue_held_rebootstrap(self) -> None:
        """Enqueue all staged rebootstrap requests when generation resumes."""
        held = self.held_rebootstrap_reqs
        self.held_rebootstrap_reqs = []
        for req in held:
            self.add(req, is_rebootstrap=True)

    @staticmethod
    def _rebootstrap_prefill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids)

    @staticmethod
    def _pre_alloc_fill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            # pause_generation(retract) already popped the boundary token out of
            # output_ids (it is replayed via the decode-side override at commit
            # time), so output_ids here is prompt + emitted-tokens-minus-boundary,
            # i.e. the original seqlen - 1. The prefill recomputes KV for *all* of
            # these tokens, leaving no just-sampled "pending" token in the list, so
            # we allocate exactly len(origin)+len(output_ids) with no -1 (unlike
            # normal decode, where the last token's KV has not been written yet).
            # This is the same token count as offloading-based retraction, where
            # offload_kv_cache saves seqlen-1 tokens; the boundary token's KV is
            # (re)computed on the decode side once generation resumes.
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        message = None
        allocator = self.token_to_kv_pool_allocator
        full_required, swa_required = self._prealloc_required_tokens(req)
        if not self._uses_swa_tail_prealloc():
            swa_required = full_required
        ceiling_fits = allocator.prealloc_ceiling_fits(full_required, swa_required)
        if ceiling_fits is False:
            message = (
                f"Request {req.rid} exceeds the unified FULL/SWA KV byte "
                f"budget: full={full_required}, swa={swa_required}"
            )
        elif ceiling_fits is None:
            # HiSparse admits up to the host-backed logical capacity.
            capacity = (
                self.scheduler.tp_worker.model_runner.max_token_pool_size
                if self.scheduler.enable_hisparse
                else self.max_total_num_tokens
            )
            input_len = self._rebootstrap_prefill_len(req)
            if input_len > capacity:
                message = f"Request {req.rid} exceeds the maximum number of tokens: {input_len} > {capacity}"
            elif self._uses_swa_tail_prealloc():
                _, swa_required = self._prealloc_required_tokens(req)
                swa_capacity = self.token_to_kv_pool_allocator.size_swa
                if swa_required > swa_capacity:
                    message = (
                        f"Request {req.rid} requires too many SWA KV tokens for "
                        f"decode preallocation: {swa_required} > {swa_capacity}"
                    )
        if message is not None:
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def release_memory_occupation(self):
        self._cancel_prefill_dp_rank_queries()
        self.queue.clear()
        for req in self.retracted_queue:
            discard_kv_cache_backup(
                req,
                self.tree_cache,
                get_disagg().disaggregation_decode_retraction_backup,
            )
        self.retracted_queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        swa_allocatable_tokens = None
        if self._uses_swa_reservation():
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(count_retracted=False)
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False
            )

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if not ensure_mamba_capacity(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator.state,
                [req],
                self.tree_cache,
            ):
                break

            full_required, swa_required = self._prealloc_required_tokens(req)
            if not self._prealloc_reservation_fits(
                full_required,
                swa_required,
                full_allocatable_tokens=full_allocatable_tokens,
                swa_allocatable_tokens=swa_allocatable_tokens,
            ):
                break

            if self.token_to_kv_pool_allocator.prealloc_fits_assumes_reclaim():
                full_len, swa_len = self._prealloc_kv_lens(req)
                if (
                    self._reclaim_swa_tail_capacity(swa_len, req.rid, full_len=full_len)
                    is not None
                ):
                    break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False,
                extra_reserved_reqs=len(resumed_reqs),
            )
            if swa_allocatable_tokens is not None:
                swa_allocatable_tokens = self._swa_tail_allocatable_token_budget(
                    count_retracted=False,
                    extra_reserved_reqs=len(resumed_reqs),
                )

            if self.scheduler.enable_overlap:
                # A request that finished last step can still have its forward in
                # flight, writing slots this request was just given.
                self.scheduler.schedule_stream.wait_stream(
                    self.scheduler.forward_stream
                )
            restore_kv_cache(
                req,
                self.tree_cache,
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                get_disagg().disaggregation_decode_retraction_backup,
            )

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _init_receiver(self, decode_req: DecodeRequest, prefill_dp_rank: int) -> None:
        decode_req.kv_receiver.init(prefill_dp_rank)
        if (
            get_disagg().disaggregation_decode_allocation_policy == "prefill_complete"
            and decode_req.kv_receiver.conclude_state != KVPoll.Failed
        ):
            decode_req.req.disagg_prefill_dp_rank = prefill_dp_rank
            # Deferred allocation cannot precede the recompute that makes its
            # source ready. Keep the existing leader election and error path.
            if decode_req.is_rebootstrap:
                self._submit_prefill_recompute(decode_req)

    def _submit_prefill_recompute(self, decode_req: DecodeRequest) -> None:
        # A fake transfer has no prefill; its receiver completes once metadata is sent.
        if _is_fake_transfer(decode_req.req):
            return
        self.kv_manager.submit_prefill_recompute(
            decode_req.kv_receiver, decode_req.req.build_rebootstrap_payload()
        )

    def _update_handshake_waiters(
        self,
        rids_to_check: Optional[List[str]] = None,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> None:
        if not self.queue:
            return

        # Receiver polling observes asynchronous failures while KV allocation
        # is blocked.
        if self.pp_size > 1:
            polls = poll_and_all_reduce_pp(
                (decode_req.req.rid for decode_req in self.queue),
                KVPoll.WaitingForInput,
                pp_good_rids,
                pp_bad_rids,
            )
        else:
            polls = poll_and_all_reduce(
                [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
            )

        for decode_req, poll in zip(self.queue, polls):
            if poll is None:
                continue
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                is_propagated = False
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    # kv_receiver may be None from a prior self.queue cleanup
                    if decode_req.kv_receiver is not None:
                        decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def prefetch_prefill_dp_rank_queries(self) -> None:
        """Start DP-rank lookups before their normal consume point."""
        if not self.pending_reqs:
            return

        queries = self._prefill_dp_rank_queries

        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        for bootstrap_addr in set(queries) - set(addr_to_reqs):
            _, stale_future = queries.pop(bootstrap_addr)
            stale_future.cancel()

        for bootstrap_addr, decode_reqs in addr_to_reqs.items():
            if bootstrap_addr in queries:
                continue
            if self.kv_manager.prefill_info_table.get(bootstrap_addr) is None:
                continue

            rooms = tuple(
                decode_req.req.bootstrap_room
                for decode_req in decode_reqs
                if self._resolve_prefill_dp_rank(decode_req.req) is None
            )
            if not rooms:
                continue

            future = self.kv_manager._ensure_prefill_recompute_executor().submit(
                CommonKVReceiver.query_prefill_dp_ranks,
                bootstrap_addr,
                list(rooms),
            )
            queries[bootstrap_addr] = (rooms, future)

    def _cancel_prefill_dp_rank_queries(self) -> None:
        for _, future in self._prefill_dp_rank_queries.values():
            future.cancel()
        self._prefill_dp_rank_queries.clear()

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            self._cancel_prefill_dp_rank_queries()
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                prefetched = self._prefill_dp_rank_queries.pop(bootstrap_addr, None)
                prefetched_rooms = prefetched[0] if prefetched is not None else ()
                if (
                    prefetched is not None
                    and tuple(rooms[: len(prefetched_rooms)]) == prefetched_rooms
                ):
                    room_to_rank = prefetched[1].result()
                    remaining_rooms = rooms[len(prefetched_rooms) :]
                    if remaining_rooms:
                        room_to_rank.update(
                            CommonKVReceiver.query_prefill_dp_ranks(
                                bootstrap_addr, remaining_rooms
                            )
                        )
                else:
                    if prefetched is not None:
                        prefetched[1].cancel()
                    room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                        bootstrap_addr, rooms
                    )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)
            else:
                prefetched = self._prefill_dp_rank_queries.pop(bootstrap_addr, None)
                if prefetched is not None:
                    prefetched[1].cancel()

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            self._init_receiver(decode_req, prefill_dp_rank)

    def pop_preallocated(
        self,
        rids_to_check: Optional[List[str]] = None,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        is_pp_mode = self.pp_size > 1
        if is_pp_mode and (pp_good_rids is None or pp_bad_rids is None):
            raise ValueError("PP consensus is required when pp_size > 1")
        if is_pp_mode and rids_to_check is not None:
            raise ValueError("rids_to_check cannot be used in PP mode")

        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check, pp_good_rids, pp_bad_rids)
        if is_pp_mode:
            rids_to_check = set(pp_good_rids) | set(pp_bad_rids)

        failed_reqs = []
        preallocated_reqs = []
        num_device_preallocated = 0
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )

        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        swa_allocatable_tokens = None
        if self._uses_swa_reservation():
            retractable_swa_tokens = sum(
                self._swa_retractable_len(r) for r in self.scheduler.running_batch.reqs
            )
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                )
            )
        else:
            retractable_swa_tokens = 0
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens, count_retracted=True
            )
        reserved_restore_tokens = self._hicache_pending_restore_tokens()
        full_allocatable_tokens -= reserved_restore_tokens
        # Sort by priority before any index-based bookkeeping so that both the
        # abort-scan loop and the preallocation loop operate on the same order.
        if self.scheduler.enable_priority_scheduling:
            priority_sign = (
                1 if self.scheduler.schedule_low_priority_values_first else -1
            )
            self.queue.sort(key=lambda r: r.req.priority * priority_sign)

        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                if not getattr(decode_req.req, "finished_output", False):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # DecodeRequest is shared between self.queue and self.pending_reqs;
        # drop failed reqs from both
        if failed_reqs:
            failed_ids = {id(r) for r in failed_reqs}
            self.pending_reqs = [
                r for r in self.pending_reqs if id(r) not in failed_ids
            ]

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs a padded buffer plus its speculative scratch.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        hisparse_req_budget = float("inf")
        if self.scheduler.enable_hisparse:
            coordinator = self.scheduler.hisparse_coordinator
            per_req_device_tokens = (
                coordinator.padded_buffer_size + coordinator.spec_swap.scratch_capacity
            )
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            hisparse_req_budget = max(
                0,
                hisparse_avail // per_req_device_tokens
                - len(self.transfer_queue.queue),
            )

        if self.scheduler.enable_lora:
            running_batches = (
                self.scheduler.running_mbs
                if is_pp_mode
                else (self.scheduler.running_batch,)
            )
            # Include finished requests; GPU work may still use their adapters.
            running_loras = {
                req.lora_id for batch in running_batches for req in batch.reqs
            }
            running_loras.update(r.req.lora_id for r in self.transfer_queue.queue)
            running_loras.update(req.lora_id for req in self.scheduler.waiting_queue)

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            if self.scheduler.enable_lora and not self.scheduler.can_schedule_lora_req(
                decode_req.req, running_loras
            ):
                continue

            if (
                get_disagg().disaggregation_decode_host_receive_threshold < 1
                and not decode_req.is_rebootstrap
                and not _is_fake_transfer(decode_req.req)
                and decode_req.kv_receiver.supports_host_destination
                and self.scheduler.pool_stats_observer.get_pool_stats().full_token_usage
                >= get_disagg().disaggregation_decode_host_receive_threshold
            ):
                if not self._pre_alloc_host(decode_req):
                    break
                preallocated_reqs.append(decode_req)
                indices_to_remove.add(i)
                if self.scheduler.enable_lora:
                    running_loras.add(decode_req.req.lora_id)
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            # Reserve enough Mamba capacity for radix COW and tracking buffers
            # before prefix matching can bind state or lock cached entries.
            if not ensure_mamba_capacity(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator.state,
                [decode_req.req],
                self.tree_cache,
            ):
                break

            if hisparse_req_budget <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = self._rebootstrap_prefill_len(decode_req.req)
            prefix_match: Optional[DecodePrefixMatch] = None
            use_decode_radix_cache = (
                get_disagg().disaggregation_decode_enable_radix_cache
                and not decode_req.is_rebootstrap
            )
            if use_decode_radix_cache:
                # Match prefix against decode's radix cache.
                prefix_match = self._match_prefix_and_lock(decode_req.req)
                prefix_indices = prefix_match.prefix_indices
                # prefix_len: tokens already on device (L1 hit).
                # total_prefix_len: full prefix promised to prefill
                # (L1 + L2 host hit + L3 storage hit), sent as PD
                # protocol's `decode_prefix_len`. The [prefix_len, total)
                # gap is filled by HiCache loadback later.
                prefix_len = prefix_match.l1_prefix_len
                total_prefix_len = prefix_match.decode_prefix_len

                fill_len = self._pre_alloc_fill_len(decode_req.req)

                # Decode transfers the SWA tail fresh, so retain only the
                # full-attention prefix lock needed for reuse.
                if (
                    uses_swa_tail_prealloc
                    and prefix_match.l1_prefix_len > 0
                    and hasattr(self.tree_cache, "release_swa")
                ):
                    self.tree_cache.release_swa(decode_req.req.lock)

                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len, prefix_len=prefix_len
                )
                # Matching may lock previously-evictable radix pages, so refresh
                # the admission budget against the post-lock pool state before we
                # decide whether this request still fits.
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=num_device_preallocated,
                    hicache_reserved_tokens=reserved_restore_tokens,
                )
                if uses_swa_tail_prealloc:
                    swa_allocatable_tokens = self._swa_tail_allocatable_token_budget(
                        retractable_tokens=retractable_tokens,
                        retractable_swa_tokens=retractable_swa_tokens,
                        count_retracted=True,
                        extra_reserved_reqs=num_device_preallocated,
                    )
            else:
                prefix_indices = None
                prefix_len = 0
                total_prefix_len = 0
                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=self._pre_alloc_fill_len(decode_req.req), prefix_len=0
                )

            full_required_for_admission = self._required_admission_tokens(
                decode_req.req, required_alloc_tokens, prefix_len, retractable_tokens
            )
            swa_required_for_admission = 0
            swa_len = required_alloc_tokens
            if uses_swa_tail_prealloc:
                _, swa_required = self._prealloc_required_tokens(decode_req.req)
                _, swa_len = self._prealloc_kv_lens(decode_req.req)
                max_new_tokens = min(
                    decode_req.req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKEN,
                )
                swa_required_for_admission = max(
                    swa_required,
                    swa_len + max_new_tokens - retractable_swa_tokens,
                )
            elif swa_allocatable_tokens is not None:
                swa_required_for_admission = full_required_for_admission

            if not self._prealloc_reservation_fits(
                full_required_for_admission,
                swa_required_for_admission,
                full_allocatable_tokens=full_allocatable_tokens,
                swa_allocatable_tokens=swa_allocatable_tokens,
            ):
                if prefix_match is not None and prefix_match.l1_prefix_len > 0:
                    self.tree_cache.unlock(decode_req.req.lock)
                    decode_req.req.lock = None
                break

            if swa_allocatable_tokens is not None:
                reclaim_error = self._reclaim_swa_tail_capacity(
                    swa_len,
                    decode_req.req.rid,
                    full_len=required_alloc_tokens,
                )
                if reclaim_error is not None:
                    if prefix_match is not None and prefix_match.l1_prefix_len > 0:
                        self.tree_cache.unlock(decode_req.req.lock)
                        decode_req.req.lock = None
                    logger.error(reclaim_error)
                    prepare_abort(
                        decode_req.req,
                        reclaim_error,
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req], decode_req.req.return_logprob
                    )
                    decode_req.kv_receiver.clear()
                    decode_req.kv_receiver = None
                    failed_reqs.append(decode_req)
                    indices_to_remove.add(i)
                    continue
            if self.scheduler.enable_decode_hicache:
                # Prefill is promised only the L3 span the prefetch took.
                self._start_hicache_prefetch(decode_req.req, prefix_match)
                if prefix_match is not None:
                    total_prefix_len = prefix_match.decode_prefix_len
            dst_kv_indices = self._pre_alloc(
                decode_req.req,
                prefix_indices,
                prefix_len,
                total_prefix_len,
            )
            decode_req.prefix_match = prefix_match
            hisparse_req_budget -= 1
            # Recompute from actual pool state for the next queue entry.
            # This accounts for page rounding and newly locked evictable cache.
            if prefix_match is not None:
                reserved_restore_tokens += prefix_match.restore_token_count
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                extra_reserved_reqs=num_device_preallocated + 1,
                hicache_reserved_tokens=reserved_restore_tokens,
            )
            if swa_allocatable_tokens is not None:
                swa_allocatable_tokens = self._swa_tail_allocatable_token_budget(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=num_device_preallocated + 1,
                )
            decode_req.req.kv.cache_protected_len = total_prefix_len

            page_size = self.token_to_kv_pool_allocator.page_size
            kv_transfer_page_size = page_size
            raw_kv_indices = self.req_to_token_pool.req_to_token[
                decode_req.req.kv.req_pool_idx
            ][total_prefix_len:origin_input_len]
            if self.scheduler.enable_hisparse:
                # Direct-to-host sends host/C4 rows; keep allocator.page_size
                # logical and use the compressed page size only for these indices.
                kv_transfer_page_size = getattr(
                    self.token_to_kv_pool_allocator,
                    "hisparse_page_size",
                    page_size,
                )
                kv_indices = dst_kv_indices[: origin_input_len - prefix_len]
            else:
                # Only send delta indices (beyond prefix) to prefill.
                kv_indices = (
                    self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                        raw_kv_indices
                    )
                )

            seq_len = origin_input_len

            def _mamba_payload():
                state_ids = self.token_to_kv_pool_allocator.state_ids(
                    self.req_to_token_pool.state_sub_pool
                )
                return [
                    state_ids.ids(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            decode_req.req.kv.req_pool_idx
                        ]
                    )
                    .cpu()
                    .numpy()
                ]

            def _swa_payload(sub_pool="swa", window=self.scheduler.sliding_window_size):
                storage = kv_storage_of(
                    self.req_to_token_pool, self.token_to_kv_pool_allocator
                )
                resume = storage.resume_units(
                    sub_pool,
                    decode_req.req,
                    token_start=total_prefix_len,
                    token_boundary=seq_len,
                    window=window,
                )
                return resume.unit_ids.cpu().numpy()

            def _full_kv_pages_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.kv.req_pool_idx, :seq_len
                ]
                # Indexer lives on device pool; always use device page_size
                device_page_size = self.token_to_kv_pool.page_size
                return kv_to_page_indices(kv_indices_full, device_page_size)

            def _minimax_payload():
                return minimax_index_transfer_pages(
                    self.req_to_token_pool.req_to_token[
                        decode_req.req.kv.req_pool_idx, :seq_len
                    ],
                    page_size=self.token_to_kv_pool.page_size,
                    prefix_len=total_prefix_len,
                )

            def _dsa_pages_payload():
                # Mirror of prefill _dsa_pages_payload: the cached prefix keeps
                # its own indexer rows, and rows still being restored by
                # HiCache are not written until the restore commits.
                return suffix_page_indices(
                    self.req_to_token_pool.req_to_token,
                    decode_req.req.kv.req_pool_idx,
                    total_prefix_len,
                    seq_len,
                    self.token_to_kv_pool.page_size,
                )

            def _dsa_tail_payload():
                return get_dsa_tail_state_indices(
                    self.token_to_kv_pool,
                    decode_req.req.kv.req_pool_idx,
                    seq_len,
                )

            def _qsa_pending_payload():
                # Match the prefill request-pool row positionally; the two
                # req_pool_idx values need not be equal.
                return get_qsa_pending_state_indices(decode_req.req)

            def _swa_ring_payload():
                # Mirror of prefill _swa_ring_payload using this side's req_pool_idx.
                # Same window positions and order -> positional match with prefill.
                ring_stride = self.token_to_kv_pool.unified_swa_ring_size
                window_size = self.token_to_kv_pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(decode_req.req.kv.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _request_state_payload():
                return self.token_to_kv_pool.request_state_transfer_indices(
                    int(decode_req.req.kv.req_pool_idx), seq_len
                )

            def _request_row_payload():
                return np.array([decode_req.req.kv.req_pool_idx], dtype=np.int32)

            state_types = self.kv_manager.kv_args.state_types
            if StateType.DSV4_REQUEST_STATE in state_types:
                clear_request_state = getattr(
                    self.token_to_kv_pool, "clear_request_scoped_state", None
                )
                if clear_request_state is not None:
                    clear_request_state(int(decode_req.req.kv.req_pool_idx))
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.QSA_PENDING: _qsa_pending_payload,
                StateType.QSA_COMPRESSED: _full_kv_pages_payload,
                StateType.SWA: _swa_payload,
                StateType.AUXILIARY_SWA: lambda: _swa_payload("draft_swa", None),
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
            if _is_npu and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )

                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        decode_req.req.kv.req_pool_idx,
                        seq_len,
                        self.token_to_kv_pool_allocator.page_size,
                        prefix_len=total_prefix_len,
                    )
                )
            state_indices: Optional[List] = [
                payloads[st]() if st in payloads else None for st in state_types
            ]

            device_page_indices = None
            if (
                self.scheduler.enable_hisparse
                and (
                    isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
                    or self.draft_token_to_kv_pool is not None
                )
                and not _is_fake_transfer(decode_req.req)
            ):
                # alloc_logical_only() already allocated the shared logical pages
                # used by the indexer and draft KV. These device buffers do not
                # use target sparse physical slots; carry logical page IDs beside
                # the independently allocated target host page IDs.
                full_kv_indices = self.req_to_token_pool.req_to_token[
                    decode_req.req.kv.req_pool_idx,
                    prefix_len:origin_input_len,
                ]
                device_page_indices = kv_to_page_indices(
                    full_kv_indices,
                    page_size,
                ).astype(np.int32)
                if self.transfer_backend != TransferBackend.MOONCAKE:
                    raise NotImplementedError(
                        "HiSparse device-only PD transfer currently requires the "
                        "Mooncake backend"
                    )
            metadata_kwargs = {"decode_prefix_len": total_prefix_len}
            if device_page_indices is not None:
                metadata_kwargs["device_kv_indices"] = device_page_indices
            self._send_kv_metadata(
                decode_req,
                kv_indices,
                kv_transfer_page_size,
                state_indices,
                **metadata_kwargs,
            )
            num_device_preallocated += 1
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            if self.scheduler.enable_lora:
                running_loras.add(decode_req.req.lora_id)

        if failed_reqs:
            failed_ids = {id(r) for r in failed_reqs}
            self.pending_reqs = [
                r for r in self.pending_reqs if id(r) not in failed_ids
            ]
            for decode_req in failed_reqs:
                if decode_req.req.kv.holds_mamba and not decode_req.req.kv.holds_kv:
                    release_kv_cache(decode_req.req, self.tree_cache, checkpoint=False)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    def _send_kv_metadata(
        self, decode_req, kv_indices, page_size, state_indices=None, **metadata_kwargs
    ) -> None:
        decode_req.metadata_buffer_index = (
            self.req_to_metadata_buffer_idx_allocator.alloc()
        )
        assert decode_req.metadata_buffer_index is not None
        page_indices = self._transfer_page_indices(decode_req, kv_indices, page_size)
        if (
            metadata_kwargs.get("destination") != KVTransferDestination.HOST
            and self.transfer_queue.enable_staging
            and hasattr(decode_req.kv_receiver, "require_staging")
            and decode_req.kv_receiver.require_staging
        ):
            # Register before send_metadata, which triggers the STAGING_REQ
            # prefetch (dropped for an unregistered room); tiny race, correct order.
            self.transfer_queue.staging_handler.register_decode_req(
                decode_req.req.bootstrap_room, decode_req
            )
        decode_req.kv_receiver.send_metadata(
            page_indices,
            decode_req.metadata_buffer_index,
            state_indices,
            **metadata_kwargs,
        )
        if (
            decode_req.is_rebootstrap
            and get_disagg().disaggregation_decode_allocation_policy == "early"
        ):
            self._submit_prefill_recompute(decode_req)
        self._num_published_destinations += 1
        decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

    def _pre_alloc_host(self, decode_req: DecodeRequest) -> bool:
        seq_len = len(decode_req.req.origin_input_ids)
        page_size = self.host_pool.page_size
        num_tokens = ceil_align(seq_len, page_size)
        required_tokens = num_tokens + self.host_reserved_tokens
        if (
            self.host_pool.available_size() < required_tokens
            and not self.tree_cache.disable
        ):
            self.tree_cache.evict_host(required_tokens)
        if self.host_pool.available_size() < required_tokens:
            return False
        swa_num_tokens = 0
        if self.host_swa_pool is not None:
            window_start = page_align_floor(
                max(0, seq_len - self.scheduler.sliding_window_size), page_size
            )
            swa_num_tokens = num_tokens - window_start
            required_swa_tokens = swa_num_tokens + self.host_swa_reserved_tokens
            if self.host_swa_pool.available_size() < required_swa_tokens:
                entry = self.tree_cache.host_pool_group.get_entry(PoolName.SWA)
                if entry.host_evict_fn is not None:
                    entry.host_evict_fn(required_swa_tokens)
            if self.host_swa_pool.available_size() < required_swa_tokens:
                return False
        host_indices = self.host_pool.alloc(num_tokens)
        if host_indices is None:
            return False
        indices_by_pool = {PoolName.KV: host_indices}
        transfers = []
        state_indices = None
        if self.host_swa_pool is not None:
            swa_host_indices = self.host_swa_pool.alloc(swa_num_tokens)
            if swa_host_indices is None:
                self.host_pool.free(host_indices)
                return False
            indices_by_pool[PoolName.SWA] = swa_host_indices
            transfers.append(
                PoolTransfer(name=PoolName.SWA, host_indices=swa_host_indices)
            )
            state_indices = [kv_to_page_indices(swa_host_indices, page_size)]
        assert decode_req.req.kv.retraction_backup is None
        decode_req.req.kv.retraction_backup = RetractionBackup(
            host_indices=host_indices,
            pool_transfers=transfers
            + [
                PoolTransfer(
                    name=spec.pool_name,
                    host_indices=indices_by_pool[spec.indices_from_pool],
                    indices_from_pool=spec.indices_from_pool,
                    hit_policy=spec.hit_policy,
                )
                for spec in self.tree_cache.sidecar_pool_specs
                if num_tokens
            ]
            or None,
        )
        if get_disagg().disaggregation_decode_enable_radix_cache:
            _bind_root_prefix(decode_req.req, self.tree_cache)
        self._send_kv_metadata(
            decode_req,
            host_indices,
            self.token_to_kv_pool_allocator.page_size,
            state_indices,
            decode_prefix_len=0,
            destination=KVTransferDestination.HOST,
        )
        decode_req.host_staged = True
        if self.scheduler.metrics_reporter.current_scheduler_metrics_enabled:
            self.scheduler.metrics_collector.increment_decode_host_receive_reqs()
        return True

    def allocate_host_staged(self, decode_req: DecodeRequest) -> bool:
        if self.req_to_token_pool.available_size() <= 0:
            return False
        req = decode_req.req
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        required_tokens = self._required_admission_tokens(
            req,
            self._required_alloc_tokens(
                fill_len=len(req.origin_input_ids), prefix_len=0
            ),
            0,
            retractable_tokens,
        )
        swa_required_tokens = 0
        swa_allocatable_tokens = None
        if self._uses_swa_reservation():
            _, swa_required_tokens = self._prealloc_required_tokens(req)
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=sum(
                        self._swa_retractable_len(r)
                        for r in self.scheduler.running_batch.reqs
                    ),
                    count_retracted=True,
                )
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens, count_retracted=True
            )
        if not self._prealloc_reservation_fits(
            required_tokens,
            swa_required_tokens,
            full_allocatable_tokens=full_allocatable_tokens,
            swa_allocatable_tokens=swa_allocatable_tokens,
        ):
            return False
        self._pre_alloc(req)
        decode_req.host_staged = False
        return True

    @property
    def has_published_destinations(self) -> bool:
        """Whether any destination address is visible to prefill but not yet
        protected by the transfer queue."""
        return self._num_published_destinations > 0

    def note_destinations_queued(self, count: int) -> None:
        """Hand `count` published destinations over to the transfer queue."""
        self._num_published_destinations = max(
            0, self._num_published_destinations - count
        )

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            decode_req.req.extend_end
            for decode_req in self.transfer_queue.queue
            if not decode_req.host_staged
        )

    def pre_allocated_token_counts(self) -> Tuple[int, int]:
        """(full, SWA) KV slots held by transfer-queue requests: KV reserved
        for an incoming transfer, not yet in the running batch. Pages are
        charged the way the pool stats charge a request's private KV."""
        page_size = self.token_to_kv_pool_allocator.page_size
        full_tokens = swa_tokens = 0
        for decode_req in self.transfer_queue.queue:
            if decode_req.host_staged:
                continue
            full_tokens += kv_private_tokens(decode_req.req.kv, page_size)
            swa_tokens += kv_private_swa_tokens(decode_req.req.kv, page_size)
        return full_tokens, swa_tokens

    def _need_space_for_single_req(
        self, retractable_tokens: Optional[int] = None
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        return need_space_for_single_req

    def _active_req_count(self, extra_reserved_reqs: int = 0) -> int:
        return (
            len(self.scheduler.running_batch.reqs)
            + sum(not r.host_staged for r in self.transfer_queue.queue)
            + len(self.scheduler.waiting_queue)
            + extra_reserved_reqs
        )

    def _active_reserved_tokens(
        self, n_active: Optional[int] = None, extra_reserved_reqs: int = 0
    ) -> int:
        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        return self.num_reserved_decode_tokens * n_active

    def _swa_aware_allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
    ) -> Tuple[int, int]:
        n_active = self._active_req_count()
        reserved_tokens = self._active_reserved_tokens(n_active)

        full_allocatable_tokens = self._allocatable_token_budgets(
            retractable_tokens=retractable_tokens,
            count_retracted=count_retracted,
            reserved_tokens=reserved_tokens,
        )

        return full_allocatable_tokens, self._swa_tail_allocatable_token_budget(
            retractable_tokens=retractable_tokens,
            retractable_swa_tokens=retractable_swa_tokens,
            count_retracted=count_retracted,
            n_active=n_active,
            reserved_tokens=reserved_tokens,
        )

    def _allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
        extra_reserved_reqs: int = 0,
        reserved_tokens: Optional[int] = None,
        hicache_reserved_tokens: int = 0,
    ) -> int:
        need_space_for_single_req = self._need_space_for_single_req(retractable_tokens)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(
                extra_reserved_reqs=extra_reserved_reqs
            )

        if self.scheduler.enable_hisparse:
            logical_allocator = self.token_to_kv_pool_allocator.logical_attn_allocator
            if self._uses_swa_tail_prealloc() and hasattr(
                logical_allocator, "full_available_size"
            ):
                available_size = logical_allocator.full_available_size()
            else:
                # HiSparse pre-alloc only allocates logical indices, so the
                # logical pool is the binding constraint for admission control.
                available_size = logical_allocator.available_size()
        elif self._uses_swa_reservation():
            available_size = self.token_to_kv_pool_allocator.full_available_size()
            if get_disagg().disaggregation_decode_enable_radix_cache:
                available_size += self._radix_full_evictable()
        else:
            available_size = self.token_to_kv_pool_allocator.available_size()
            # Include evictable decode-radix cache entries in the budget -- they
            # can be freed on demand before allocation.
            if get_disagg().disaggregation_decode_enable_radix_cache:
                available_size += self._radix_full_evictable()
        allocatable_tokens = available_size - max(
            reserved_tokens, need_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            for req in self.retracted_queue:
                full_required, _ = self._prealloc_required_tokens(req)
                allocatable_tokens -= full_required

        allocatable_tokens -= hicache_reserved_tokens
        return allocatable_tokens

    def _swa_tail_allocatable_token_budget(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        n_active: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
        extra_reserved_reqs: int = 0,
    ) -> int:
        need_swa_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens
        )
        if (
            retractable_swa_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            need_swa_space_for_single_req = max(
                self._swa_tail_len(len(x.origin_input_ids))
                + min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                - retractable_swa_tokens
                for x in self.scheduler.running_batch.reqs
            )

        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(n_active)

        # SWA growth is bounded by the sliding window: once a req's SWA
        # footprint reaches `sliding_window_size`, further decode tokens
        # evict old ones and net growth is zero. The linear reservation
        # `num_reserved_decode_tokens * n_active` (correct for the full
        # pool) over-reserves SWA in steady state. Cap by the actual
        # remaining headroom up to per-req window cap.
        window_size = self.scheduler.sliding_window_size or 0
        allocator = self.token_to_kv_pool_allocator
        _, (swa_total, swa_available) = allocator.swa_capacity_and_available(
            full_capacity=allocator.size_full, swa_capacity=allocator.size_swa
        )
        # Per-request SWA ring: cached prefixes still report swa_evictable, but
        # evicting them frees no ring space.
        swa_evictable = (
            0
            if is_swa_req_ring(self.token_to_kv_pool_allocator)
            else self.tree_cache.swa_evictable_size()
        )
        swa_used = swa_total - swa_available - swa_evictable
        swa_growth_potential = max(0, n_active * window_size - swa_used)
        swa_reserved_tokens = min(reserved_tokens, swa_growth_potential)
        swa_allocatable_tokens = (
            swa_available
            + swa_evictable
            - max(swa_reserved_tokens, need_swa_space_for_single_req)
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            prebuilt_reserved_tokens = self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )
            prebuilt_n = len(self.scheduler.last_batch.reqs)
            prebuilt_swa_growth = max(0, prebuilt_n * window_size - swa_used)
            swa_allocatable_tokens -= min(prebuilt_reserved_tokens, prebuilt_swa_growth)

        if count_retracted:
            for req in self.retracted_queue:
                _, swa_required = self._prealloc_required_tokens(req)
                swa_allocatable_tokens -= swa_required

        return swa_allocatable_tokens

    def _required_alloc_tokens(self, *, fill_len: int, prefix_len: int) -> int:
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _alloc_for_decode_prealloc(
        self, allocator: BaseTokenToKVPoolAllocator, **kwargs
    ) -> torch.Tensor:
        storage = kv_storage_of(
            self.req_to_token_pool, allocator, tree_cache=self.tree_cache
        )
        return alloc_for_decode_prealloc(storage, allocator, **kwargs)

    def _transfer_page_indices(
        self, decode_req: DecodeRequest, kv_indices: torch.Tensor, page_size: int
    ) -> np.ndarray:
        # int32 for ZMQ serialization -- from_zmq reads np.int32.
        return kv_to_page_indices(kv_indices, page_size).astype(np.int32)

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        total_prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool.

        ``prefix_len`` is the L1 device-resident prefix length (already
        backed by ``prefix_indices``). ``total_prefix_len`` is the full
        prefix committed to prefill as ``decode_prefix_len`` (L1 + L2 + L3);
        the ``[prefix_len, total_prefix_len)`` gap is filled later by HiCache
        loadback.
        """
        if prefix_len is None:
            prefix_len = 0
        if total_prefix_len is None:
            total_prefix_len = prefix_len

        req_pool_indices = kv_storage_of(
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
        ).acquire([req])

        assert req_pool_indices is not None, (
            "req_pool_indices is full! There is a bug in memory estimation."
        )

        fill_len = self._pre_alloc_fill_len(req)
        req.kv.kv_committed_len = fill_len

        if prefix_len > 0:
            self.req_to_token_pool.write(
                (req.kv.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        # TODO(retraction): when retraction is implemented with radix cache
        # awareness, a retracted request should re-match the tree here
        # instead of re-allocating from scratch. See resume_retracted_reqs.
        delta_len = fill_len - total_prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len, prefix_len=prefix_len
        )

        # Evict cached entries if the pool doesn't have enough free pages.
        if (
            get_disagg().disaggregation_decode_enable_radix_cache
            and self._radix_full_available() < required_alloc_tokens
        ):
            num_to_evict = required_alloc_tokens - self._radix_full_available()
            result = self.tree_cache.evict_for_alloc(
                EvictParams(num_tokens=num_to_evict)
            )
            if self._radix_full_available() < required_alloc_tokens:
                logger.warning(
                    f"Eviction insufficient: needed {required_alloc_tokens} tokens, "
                    f"available {self._radix_full_available()} "
                    f"after evicting {result.num_tokens_evicted}/{num_to_evict} tokens. "
                    f"evictable_size={self._radix_full_evictable()}, "
                    f"protected_size={self._radix_full_protected()}, "
                    f"fill_len={fill_len}, prefix_len={prefix_len}, "
                    f"total_prefix_len={total_prefix_len}, delta_len={delta_len}, "
                    f"page_size={self.token_to_kv_pool_allocator.page_size}, "
                    f"req={req.rid}"
                )

        allocator = self.token_to_kv_pool_allocator
        uses_swa_tail = self._uses_swa_tail_prealloc()
        swa_tail_len = self._swa_tail_len(fill_len)
        swa_pages_charged = (
            uses_swa_tail
            and not is_swa_req_ring(allocator)
            and not self.scheduler.enable_hisparse
        )
        required_swa_tokens = (
            ceil_align(swa_tail_len, allocator.page_size) if swa_pages_charged else 0
        )
        if get_disagg().disaggregation_decode_enable_radix_cache and swa_pages_charged:
            # Admission includes evictable pages; allocation needs free pages.
            # Separate-pool resumes skip the caller's unified-only reclaim.
            reclaim_error = self._reclaim_swa_tail_capacity(
                swa_tail_len, req.rid, full_len=required_alloc_tokens
            )
            if reclaim_error is not None:
                logger.warning("%s", reclaim_error)

        if self.scheduler.enable_hisparse:
            # HiSparse is incompatible with decode-side L1 radix cache. Keep
            # this path on the upstream full-allocation semantics.
            assert prefix_len == 0

            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            coordinator = self.scheduler.hisparse_coordinator
            kv_loc = alloc_for_decode_prealloc_hisparse(
                allocator,
                req=req,
                fill_len=fill_len,
                uses_swa_tail=uses_swa_tail,
                swa_tail_len=swa_tail_len,
            )
            # Allocate host indices for the RDMA transfer target.
            host_indices = coordinator.mem_pool_host.alloc_paged_token_slots(
                coordinator.req_to_host_pool,
                coordinator.req_to_host_pool_allocated_len,
                req.kv.req_pool_idx,
                0,
                coordinator.host_token_len(fill_len),
            )
        else:
            kv_loc = self._alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=fill_len,
                delta_len=delta_len,
                prefix_len=prefix_len,
                total_prefix_len=total_prefix_len,
                prefix_indices=prefix_indices,
                uses_swa_tail=uses_swa_tail,
                swa_tail_len=swa_tail_len,
                req_to_token_pool=self.req_to_token_pool,
            )
        assert kv_loc is not None, (
            f"KV cache is full! Bug in memory estimation. "
            f"available={self._radix_full_available()}, "
            f"evictable={self._radix_full_evictable()}, "
            f"protected={self._radix_full_protected()}, "
            f"required_alloc={required_alloc_tokens}, delta={delta_len}, "
            f"fill={fill_len}, prefix={prefix_len}, total_prefix={total_prefix_len}, "
            f"swa_available={allocator.swa_available_size() if swa_pages_charged else -1}, "
            f"swa_evictable={self.tree_cache.swa_evictable_size() if swa_pages_charged else -1}, "
            f"required_swa={required_swa_tokens}, swa_tail_len={swa_tail_len}, "
            f"page_size={allocator.page_size}, "
            f"req={req.rid}"
        )

        self.req_to_token_pool.write(
            (
                req.kv.req_pool_idx,
                slice(total_prefix_len, total_prefix_len + len(kv_loc)),
            ),
            kv_loc,
        )

        # Truncate fill_len to kv_committed_len so checkpoint only
        # inserts committed KV into the radix tree. The last output token
        # hasn't had KV committed yet (output_ids is 1 ahead).
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        # Decode-disagg allocates before batch assembly, so it binds the prefix
        # here instead of in init_next_round_input.
        req.prefix_len = prefix_len
        req.extend_end = req.kv.kv_committed_len
        self.tree_cache.maybe_hand_to_session(req)

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


def alloc_for_decode_prealloc_hisparse(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    uses_swa_tail: bool,
    swa_tail_len: int,
) -> torch.Tensor:
    req.kv.kv_allocated_len = fill_len
    device = allocator.device
    prefix_lens = async_h2d([0], dtype=torch.int64, device=device)
    prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
    seq_lens = async_h2d([fill_len], dtype=torch.int64, device=device)
    seq_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
    last_loc = async_h2d([-1], dtype=torch.int64, device=device)
    if uses_swa_tail:
        kv_loc = allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
            swa_tail_len=swa_tail_len,
        )
        swa_evicted_seqlen = fill_len - swa_tail_len
        assert swa_evicted_seqlen >= 0 and swa_evicted_seqlen % allocator.page_size == 0
        req.kv.set_evicted_seqlen(ComponentType.SWA, swa_evicted_seqlen)
    else:
        kv_loc = allocator.alloc_logical_only(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
        )
    return kv_loc


def _evicted_by_caller(num_tokens: int) -> None:
    """A pre-allocation's room was made before it: by the radix eviction in
    `_pre_alloc`, or by admission."""


def alloc_for_decode_prealloc(
    storage: KVStorage | StaticKVStorage,
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    delta_len: int,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
    swa_tail_len: int,
    req_to_token_pool: Optional[ReqToTokenPool] = None,
) -> torch.Tensor:
    req.kv.kv_allocated_len = fill_len
    if allocator.page_size == 1:
        return storage.take_tokens(delta_len, make_room=_evicted_by_caller)
    device = allocator.device
    last_loc = (
        prefix_indices[-1:].to(dtype=torch.int64, device=device)
        if prefix_len > 0
        else async_h2d([-1], dtype=torch.int64, device=device)
    )
    if hasattr(allocator, "c128_attn_allocator"):
        # NPU DeepSeek-V4 pre-allocates its c4/c128 pages with the tokens.
        return _alloc_for_decode_prealloc_dsv4(
            allocator,
            req=req,
            fill_len=fill_len,
            delta_len=delta_len,
            total_prefix_len=total_prefix_len,
            last_loc=last_loc,
            uses_swa_tail=uses_swa_tail,
            swa_tail_len=swa_tail_len,
            req_to_token_pool=req_to_token_pool,
        )
    kv_loc = storage.take_extend(
        rows=None,
        prefix_lens=async_h2d([total_prefix_len], dtype=torch.int64, device=device),
        prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
        seq_lens=async_h2d([fill_len], dtype=torch.int64, device=device),
        seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
        last_loc=last_loc,
        num_tokens=delta_len,
        make_room=_evicted_by_caller,
        # Full-attention layers reuse prefix KV; SWA layers allocate only the
        # live window tail.
        window_tail=swa_tail_len if uses_swa_tail else None,
    )
    if uses_swa_tail:
        swa_evicted_seqlen = fill_len - swa_tail_len
        assert swa_evicted_seqlen >= 0 and swa_evicted_seqlen % allocator.page_size == 0
        req.kv.set_evicted_seqlen(ComponentType.SWA, swa_evicted_seqlen)
    return kv_loc


def _alloc_for_decode_prealloc_dsv4(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    delta_len: int,
    total_prefix_len: int,
    last_loc: torch.Tensor,
    uses_swa_tail: bool,
    swa_tail_len: int,
    req_to_token_pool: Optional[ReqToTokenPool],
) -> torch.Tensor:
    assert req_to_token_pool is not None
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
        dsv4_prealloc_kwargs,
        dsv4_unwrap_prealloc,
    )

    device = allocator.device
    extra_kwargs = dsv4_prealloc_kwargs(
        allocator,
        req,
        fill_len,
        req_to_token_pool,
        device=device,
    )
    if uses_swa_tail:
        kv_loc = allocator.alloc_extend_swa_tail(
            prefix_lens=async_h2d([total_prefix_len], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
            seq_lens=async_h2d([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=last_loc,
            extend_num_tokens=delta_len,
            swa_tail_len=swa_tail_len,
            **extra_kwargs,
        )
        swa_evicted_seqlen = fill_len - swa_tail_len
        assert swa_evicted_seqlen >= 0 and swa_evicted_seqlen % allocator.page_size == 0
        req.kv.set_evicted_seqlen(ComponentType.SWA, swa_evicted_seqlen)
    else:
        kv_loc = allocator.alloc_extend(
            prefix_lens=async_h2d([total_prefix_len], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
            seq_lens=async_h2d([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=last_loc,
            extend_num_tokens=delta_len,
            **extra_kwargs,
        )
    return dsv4_unwrap_prealloc(
        kv_loc, req_to_token_pool, req, total_prefix_len, fill_len
    )


def _generate_fake_prefill_handoff_output_id(req: Req) -> int:
    """Generate a deterministic synthetic handoff token for fake-PD requests.

    Fake-PD skips prefill and KV transfer, so its allocated prompt KV has no
    request-specific state and its metadata output token remains zero. Reusing
    token 0 for every request can unrealistically correlate decode workloads.
    Hash the request ID so every TP rank gets the same request-diverse token.
    """
    if req.vocab_size is None or req.vocab_size <= 0:
        raise ValueError("Fake-PD requests require a positive vocabulary size")
    digest = hashlib.blake2b(
        req.rid.encode(), digest_size=8, person=b"fake-pd"
    ).digest()
    return int.from_bytes(digest, byteorder="little") % req.vocab_size


class DecodeTransferQueue(DecodeHiCacheTransferMixin):
    """
    Store the requests that is polling kv
    """

    enable_host_receive: bool = False

    def __init__(
        self,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = get_parallel().attn_tp_group.cpu_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = get_parallel().tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.enable_host_receive = (
            get_disagg().disaggregation_decode_host_receive_threshold < 1
        )
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        # Prefill layouts whose digests cannot be compared, logged once each.
        self._skipped_checksum_signatures: set = set()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None
        self.enable_deferred_kv_release = (
            envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE.get()
        )
        self.deferred_kv_release_timeout = (
            envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT.get()
        )
        # Aborted-mid-transfer requests whose KV pages/slot are held until drained
        # or timed out. Entries:
        # (decode_req, start_time, deadline, metadata_idx, required_acks).
        self._deferred_releases: List[Tuple[DecodeRequest, float, float, int, int]] = []

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)
        # This queue now covers them.
        prealloc_queue = self.scheduler.disagg_decode_prealloc_queue
        if prealloc_queue is not None:
            prealloc_queue.note_destinations_queued(len(decode_reqs))

    def _kv_checksum_comparable(self, signature: int) -> bool:
        """Digests compare only when the prefill digested this decode's layout.
        One with another TP width or a PP stage is admitted unverified and
        logged once, not aborted on every request. A prefill that wrote no
        digest (signature 0) is skipped silently; its registration already
        logged the buffer-count mismatch."""
        mine = self.scheduler.disagg_decode_prealloc_queue.kv_checksum_signature
        if signature == mine:
            return True
        if signature and signature not in self._skipped_checksum_signatures:
            self._skipped_checksum_signatures.add(signature)
            logger.warning(
                "KV checksum skipped: prefill layout signature %#x differs from "
                "this decode's %#x (TP width, PP stage, or page size); its "
                "requests are admitted unverified.",
                signature,
                mine,
            )
        return False

    def _commit_transfer_to_req(self, decode_req: DecodeRequest):
        # Keep the checksum before the metadata slot is freed; it is verified
        # when the request enters a batch.
        idx = decode_req.metadata_buffer_index
        if self.scheduler.kv_checksum_computer is not None:
            digest, signature = self.metadata_buffers.get_kv_checksum(idx)
            decode_req.req.expected_kv_checksum = (
                digest if self._kv_checksum_comparable(signature) else 0
            )
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_token_sampling_mask_len,
            output_token_sampling_mask_idx,
            output_token_sampling_logprobs,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_dsa_topk_indices,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req):
            pass
        elif actual_room == 0:
            # Should never happen: _poll_with_metadata_gate already confirmed
            # readiness on all TP ranks. Abort deterministically to avoid
            # cross-rank queue divergence.
            logger.error(
                f"Metadata unexpectedly not ready after readiness gate: "
                f"request {decode_req.req.rid}, bootstrap_room={expected_room}, "
                f"metadata_buffer_index={idx}"
            )
            prepare_abort(
                decode_req.req,
                "Metadata unexpectedly not ready after readiness gate "
                "(bootstrap_room=0)",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return
        elif actual_room != expected_room:
            # Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return

        self._commit_hicache_local_restore_to_req(decode_req)

        # Case 3: Success - commit the transfer
        # PD true-retraction rebootstrap: the prefill recomputed the prefix KV
        # under the current weights and sampled a fresh handoff token, but when
        # there is a remembered boundary token we are *replaying* an
        # already-emitted token. Override the handoff with it, and skip
        # re-committing a logprob or sampling mask for it -- it keeps its original
        # behavior logprob and sampling mask from before the retract (we never
        # re-score generated tokens under the new policy). A rebootstrap with no
        # boundary token (retracted before emitting any output) falls through to
        # the normal path so its first token and logprob are committed as usual.
        replayed_boundary = (
            decode_req.is_rebootstrap
            and decode_req.req.pd_rebootstrap_forced_output_id is not None
        )
        if replayed_boundary:
            committed_output_id = decode_req.req.pd_rebootstrap_forced_output_id
            decode_req.req.pd_rebootstrap_forced_output_id = None
        elif _is_fake_transfer(decode_req.req):
            committed_output_id = _generate_fake_prefill_handoff_output_id(
                decode_req.req
            )
        else:
            committed_output_id = output_id[0].item()
        decode_req.req.output_ids.append(committed_output_id)
        if not replayed_boundary:
            # The handoff token is generated on the prefill worker, so it does
            # not pass through the decode worker's normal batch-result path.
            # Account for it here using the same request-selected reasoning
            # terminator matcher as subsequent decode tokens.  A rebootstrap
            # boundary has already been accounted for before retraction and
            # must not be consumed twice.
            self.scheduler.batch_result_processor._maybe_update_reasoning_tokens(
                decode_req.req, committed_output_id
            )
        decode_req.req.cached_tokens = cached_tokens[0].item()
        # The prefill node already reported its prefix-cache hit in
        # cached_tokens[0]. Seed already_computed with it so that
        # prepare_for_prebuilt's `cached_tokens += pre_len - already_computed`
        # only adds decode-side reuse *beyond* what prefill counted, instead of
        # double-counting the shared prompt prefix (which would make
        # cached_tokens exceed prompt_tokens when decode radix cache is on).
        decode_req.req.already_computed = decode_req.req.cached_tokens
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        # Multimodal prompt token counts packed into cached_tokens slots 4-6
        # by the prefill node (see MetadataBuffers.set_buf).
        decode_req.req.mm_image_tokens = cached_tokens[4].item()
        decode_req.req.mm_audio_tokens = cached_tokens[5].item()
        decode_req.req.mm_video_tokens = cached_tokens[6].item()
        decode_req.req.mm_token_counts_ready = True
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states
            if (
                output_dsa_topk_indices is not None
                and torch.all(output_dsa_topk_indices < 0).item()
            ):
                output_dsa_topk_indices = None
            decode_req.req.output_dsa_topk_indices = output_dsa_topk_indices

        if decode_req.req.return_logprob and not replayed_boundary:
            decode_req.req.logprob.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.logprob.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.logprob.output_top_logprobs_val.append(
                output_top_logprobs_val[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            decode_req.req.logprob.output_top_logprobs_idx.append(
                output_top_logprobs_idx[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            if decode_req.req.logprob.token_ids_logprob is not None:
                # Prefill omits these scores; reserve a row for the streaming offset.
                decode_req.req.logprob.output_token_ids_logprobs_val.append([])
                decode_req.req.logprob.output_token_ids_logprobs_idx.append([])
        if decode_req.req.return_sampling_mask and not replayed_boundary:
            assert output_token_sampling_mask_idx is not None, (
                "sampling mask buffer disabled on decode side"
            )
            sampling_mask_len = int(output_token_sampling_mask_len[0].item())
            num_logprobs = (
                sampling_mask_len
                if decode_req.req.sampling_logprobs_mode == "support"
                else 1
            )
            decode_req.req.sampling_mask_rows.append(
                output_token_sampling_mask_idx[:sampling_mask_len].cpu().numpy(),
                output_token_sampling_logprobs[:num_logprobs].cpu().numpy(),
            )

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return

    def _pollers(self) -> list:
        if self.scheduler.enable_decode_hicache:
            return [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
        return [dr.kv_receiver for dr in self.queue]

    def _poll_with_metadata_gate(self) -> List[int]:
        return poll_and_all_reduce(
            self._pollers(),
            self.gloo_group,
            decode_reqs=self.queue,
            metadata_buffers=self.metadata_buffers,
        )

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue,
            self.staging_handler,
            self.gloo_group,
            metadata_buffers=self.metadata_buffers,
            pollers=self._pollers(),
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def _release_request(self, decode_req: DecodeRequest) -> None:
        # HiSparse host pages are the transfer destination; free them with the KV.
        if self.scheduler.enable_hisparse:
            self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
        if self.enable_host_receive:
            discard_kv_cache_backup(decode_req.req, self.tree_cache, "host_pool")
            if decode_req.host_staged:
                return
        release_kv_cache(decode_req.req, self.tree_cache, checkpoint=False)

    def _is_host_destination(self, decode_req: DecodeRequest) -> bool:
        return self.enable_host_receive and decode_req.host_staged

    def _record_transfer_failure(self, failure_reason: str) -> None:
        if self.scheduler.metrics_reporter.enable_metrics:
            self.scheduler.metrics_collector.increment_transfer_failed_reqs()

    def _abort_failed_transfer(self, decode_req: DecodeRequest) -> None:
        if self._is_host_destination(decode_req):
            decode_req.kv_receiver.abort()

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            return []

        if self.scheduler.enable_decode_hicache:
            self._process_hicache_local_restores(
                [
                    decode_req
                    for decode_req in self.queue
                    if rids_to_check is None or decode_req.req.rid in rids_to_check
                ]
            )

        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = self._poll_with_metadata_gate()

        transferred_reqs = []
        indices_to_remove = set()
        # Queue-removed but held for deferred release; excluded from the metadata
        # teardown below.
        deferred_indices = set()
        failed: List[Tuple[int, DecodeRequest, bool]] = []
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                self._abort_failed_transfer(decode_req)
                error_message = (
                    f"Decode transfer failed for request rank={self.tp_rank} "
                    f"{decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                )
                is_propagated = False
                failure_reason = "KV transfer failed"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                    failure_reason = getattr(e, "failure_reason", str(e))
                if decode_req.hicache_restore_status == HiCacheRestoreResult.FAILED:
                    is_propagated = False
                    failure_reason = "HiCache restore failed"
                    error_message += " (HiCache local restore failed on this rank)"
                self._clean_hicache_prefetch_resources(decode_req)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                requires_host_drain = self._is_host_destination(decode_req)
                deferrable = (
                    self.enable_deferred_kv_release
                    and decode_req.kv_receiver.kv_mgr.enable_deferred_decode_kv_release
                )
                if deferrable and not decode_req.kv_receiver.abort_notified:
                    # A failure decode did not initiate (prefill fault, transport
                    # error, hicache restore failure) can still have sibling-rank
                    # writes in flight toward these pages. force_arm: a partial
                    # send_metadata leaves init_time None though earlier ranks
                    # already hold destinations; this path always reaches
                    # _do_release, which clears the tracker.
                    decode_req.kv_receiver.ensure_abort_notified(force_arm=True)
                failed.append(
                    (
                        i,
                        decode_req,
                        requires_host_drain
                        or (deferrable and decode_req.kv_receiver.abort_notified),
                    )
                )
                indices_to_remove.add(i)
                self._record_transfer_failure(failure_reason)
                continue
            elif poll == KVPoll.Success:
                if self._is_host_destination(decode_req):
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        self.scheduler.output_streamer.stream_output(
                            [decode_req.req], decode_req.req.return_logprob
                        )
                        self._release_request(decode_req)
                        decode_req.kv_receiver.clear()
                        decode_req.kv_receiver = None
                        indices_to_remove.add(i)
                        continue
                    if not self.scheduler.disagg_decode_prealloc_queue.allocate_host_staged(
                        decode_req
                    ):
                        continue
                self._commit_transfer_to_req(decode_req)
                indices_to_remove.add(i)
                # Check if request was aborted due to corruption
                if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                    self._clean_hicache_prefetch_resources(decode_req)
                    self._release_request(decode_req)
                    self._record_transfer_failure(
                        decode_req.req.finished_reason.message
                    )
                else:
                    transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        # Notification state is rank-local. If any rank must hold a failed
        # transfer, all must retain its capacity so admission stays identical.
        held = all_reduce_flags(
            flags=[hold for _, _, hold in failed],
            group=self.gloo_group,
            op=torch.distributed.ReduceOp.MAX,
        )
        for (i, decode_req, _), hold in zip(failed, held):
            if hold:
                self._defer_release(decode_req)
                deferred_indices.add(i)
            else:
                self._release_request(decode_req)
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None

        for i in indices_to_remove:
            if i in deferred_indices:
                # Held for deferred release; metadata buffer freed at resolve time.
                continue
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            # Reset so the next owner sees actual_room == 0 ("not yet written")
            # instead of the stale value, avoiding a false-positive mismatch.
            self.metadata_buffers.clear_slot(idx)
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs

    def _defer_release(self, decode_req: DecodeRequest) -> None:
        start_time = time.monotonic()
        deadline = start_time + self.deferred_kv_release_timeout
        # Host-capable receivers own their drain tracking. Device destinations
        # snapshot the number of prefill acknowledgements required by the manager.
        # A rank that only holds for a peer has no local acknowledgements to await.
        required_acks = (
            0
            if self._is_host_destination(decode_req)
            or not decode_req.kv_receiver.abort_notified
            else len(decode_req.kv_receiver.bootstrap_infos)
        )
        self._deferred_releases.append(
            (
                decode_req,
                start_time,
                deadline,
                decode_req.metadata_buffer_index,
                required_acks,
            )
        )

    def _do_release(self, decode_req: DecodeRequest, idx: int) -> None:
        room = decode_req.req.bootstrap_room
        if self.enable_staging and self.staging_handler.is_staging_room(room):
            self.staging_handler.unregister_decode_req(room)
        # release pre-allocated kv cache, but don't insert into the tree since it's failed
        self._release_request(decode_req)
        self.metadata_buffers.clear_slot(idx)
        self.req_to_metadata_buffer_idx_allocator.free(idx)
        if not self._is_host_destination(decode_req):
            decode_req.kv_receiver.kv_mgr.clear_deferred_abort_state(room)
        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None

    def has_pending_deferred_releases(self) -> bool:
        return bool(self._deferred_releases)

    def num_pending_deferred_releases(self) -> int:
        return len(self._deferred_releases)

    def _observe_deferred_kv_release(
        self, duration_seconds: float, outcome: str
    ) -> None:
        if self.scheduler.metrics_reporter.enable_metrics:
            self.scheduler.metrics_collector.observe_decode_deferred_kv_release(
                duration_seconds=duration_seconds,
                outcome=outcome,
            )

    def resolve_deferred_releases(self) -> None:
        """Release drained transfers; device destinations may also time out."""
        if not self._deferred_releases:
            return
        now = time.monotonic()
        drained = []
        ready = []
        for decode_req, _, deadline, _, required_acks in self._deferred_releases:
            if self._is_host_destination(decode_req):
                # A timeout cannot prove that a remote write has stopped.
                is_drained = decode_req.kv_receiver.is_abort_release_safe()
                is_ready = is_drained
            else:
                is_drained = decode_req.kv_receiver.kv_mgr.is_abort_release_safe(
                    decode_req.req.bootstrap_room, required_acks
                )
                is_ready = is_drained or now >= deadline
            drained.append(is_drained)
            ready.append(is_ready)
        # Acks and deadlines are rank-local. Releasing only on some ranks would
        # diverge admission and the sizes of subsequent polling collectives.
        # Entries are appended from agreed holds, so every rank reduces the same
        # list in the same order.
        ready = all_reduce_flags(
            flags=ready, group=self.gloo_group, op=torch.distributed.ReduceOp.MIN
        )
        still_held = []
        to_release = []
        for entry, is_ready, is_drained in zip(self._deferred_releases, ready, drained):
            decode_req, start_time, _, idx, _ = entry
            if is_ready:
                to_release.append(
                    (
                        decode_req,
                        start_time,
                        idx,
                        decode_req.req.bootstrap_room,
                        is_drained,
                    )
                )
            else:
                still_held.append(entry)
        # Commit the survivors before releasing so a _do_release exception can't
        # leave a released entry in the list (double-free / None receiver on retry).
        self._deferred_releases = still_held
        for decode_req, start_time, idx, room, drained in to_release:
            if not drained:
                logger.warning(
                    f"Deferred KV release for room {room} timed out after "
                    f"{self.deferred_kv_release_timeout}s without a full drain "
                    f"ack from prefill; releasing anyway."
                )
            outcome = "drained" if drained else "timeout"
            try:
                self._do_release(decode_req, idx)
            except Exception:
                # Isolate a failed release so the rest still run; entry already dropped.
                logger.exception(f"Deferred KV release failed for room {room}")
                outcome = "error"
            self._observe_deferred_kv_release(
                duration_seconds=max(0.0, now - start_time),
                outcome=outcome,
            )

    def release_memory_occupation(self):
        """Clean up in-flight transfers before releasing GPU memory."""
        self.queue.clear()
        # Pool is being torn down; drop held entries without per-request release.
        self._deferred_releases.clear()

    def resume_memory_occupation(self):
        """Queues are already cleared on release; new transfers can be accepted."""
        pass


class SchedulerDisaggregationDecodeMixin:
    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            if self.gracefully_exit:
                break

            # Pending rooms from the prior cycle can overlap request intake and
            # the tail of the in-flight decode graph.
            if not self._engine_paused:
                self.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries()
            # Receive requests
            self.ingest_requests()
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self._sched_idled = True
                self.on_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            if self.gracefully_exit:
                break

            # Pending rooms from the prior cycle can overlap request intake and
            # the tail of the in-flight decode graph.
            if not self._engine_paused:
                self.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries()
            # Receive requests
            self.ingest_requests()
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self._sched_idled = True

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch(running_batch)
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if running_batch.is_empty():
                    running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        running_batch.hisparse_coordinator = self.hisparse_coordinator
                else:
                    running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if running_batch.is_empty():
            ret = None
        else:
            running_batch = self.update_running_batch(running_batch)
            ret = running_batch if not running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def get_new_prebuilt_batch(
        self, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        batch = self._get_new_prebuilt_batch(running_batch)
        if batch is None:
            return None
        # Verify after host restore and before process_prebuilt caches any KV.
        self._verify_kv_checksums(batch)
        if batch.is_empty():
            return None
        self.ngram_embedding_manager.prepare_for_forward(
            batch, chunked_req=self.chunked_req
        )
        batch.process_prebuilt(self.future_map)
        return batch

    def _verify_kv_checksums(self, batch: ScheduleBatch) -> None:
        computer: Optional[KvChecksumComputer] = self.kv_checksum_computer
        if computer is None:
            return

        prealloc_queue = self.disagg_decode_prealloc_queue
        kv_manager = prealloc_queue.kv_manager
        mismatch_prob = envs.SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB.get()
        mismatched = [0] * len(batch.reqs)
        messages: Dict[int, str] = {}
        for i, req in enumerate(batch.reqs):
            expected = req.expected_kv_checksum
            if (
                expected == 0
                or is_health_check_req(req)
                or not kv_manager.should_verify_kv_checksum(_bootstrap_addr(req))
            ):
                continue
            # Verified once. A retraction restores the row after decode has
            # written its own tokens into the last page, so the handoff digest
            # no longer describes it.
            req.expected_kv_checksum = 0
            seq_len = len(req.origin_input_ids)
            # The tree cache owns [0, cache_protected_len): the prefix this
            # decode promised at bootstrap, so prefill sent and digested only
            # what follows it.
            start_idx = req.kv.cache_protected_len
            page_indices_gpu = page_indices_for_request(
                self, req, seq_len, start_idx=start_idx
            )
            state_indices = state_indices_for_request(
                self, req, seq_len, start_idx=start_idx
            )
            actual = computer.compute(page_indices_gpu, state_indices)
            if (
                mismatch_prob > 0
                and prealloc_queue.kv_checksum_mismatch_injector.random()
                < mismatch_prob
            ):
                actual ^= 1
            if actual != expected:
                mismatched[i] = 1
                messages[i] = (
                    f"KV checksum mismatch req={req.rid} "
                    f"bootstrap_room={req.bootstrap_room} "
                    f"expected={expected:#x} got={actual:#x}"
                )

        # Corruption is per rank. Agree on the drop set first, or the TP ranks
        # run different batches and hang on the next collective.
        mismatched = all_reduce_flags(
            mismatched, self.attn_tp_cpu_group, dist.ReduceOp.MAX
        )
        if not any(mismatched):
            return
        for i, req in enumerate(batch.reqs):
            if mismatched[i]:
                msg = messages.get(
                    i,
                    f"KV checksum mismatch on a peer rank req={req.rid} "
                    f"bootstrap_room={req.bootstrap_room}",
                )
                logger.error(msg)
                self._handle_kv_checksum_mismatch(req, msg)
        batch.filter_batch(
            keep_indices=[i for i, bad in enumerate(mismatched) if not bad]
        )

    def _handle_kv_checksum_mismatch(self, req: Req, msg: str) -> None:
        # A mismatch means the KV this worker received is not what prefill sent,
        # so the cause is hardware or transport rather than the request. Serving
        # keeps going and drops just this request; CI fails instead, because a
        # single aborted request is easy to miss in a passing run -- unless the
        # mismatch was injected to exercise this very path.
        if is_in_ci() and envs.SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB.get() == 0:
            raise RuntimeError(msg)
        prepare_abort(
            req,
            "KV checksum mismatch",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        self.output_streamer.stream_output([req], req.return_logprob)
        release_kv_cache(req, self.tree_cache, checkpoint=False)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()

    def _get_new_prebuilt_batch(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        if self.enable_priority_scheduling:
            self.policy.calc_priority(self.waiting_queue, running_batch)

        host_restore = get_disagg().disaggregation_decode_host_receive_threshold < 1
        restores_ready = set()
        if host_restore:
            self._start_host_restores()
            restores_ready = self._host_restores_ready()

        curr_batch_size = running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for req in self.waiting_queue:
            # we can only add at least `num_not_used_batch` new batch to the running queue;
            # a request restored from host memory waits until its copy has finished
            if len(can_run_list) < num_not_used_batch and (
                not host_restore
                or req.kv.restore_done_event is None
                or id(req) in restores_ready
            ):
                can_run_list.append(req)
                if req.kv.restore_done_event is not None:
                    finish_restore_kv_cache(req, self.tree_cache)
                # `pop_preallocated` matched and locked new requests; a retracted or
                # rebootstrapped one owns its row, and a re-match here takes no lock.
                if get_disagg().disaggregation_decode_enable_radix_cache:
                    if req.last_node is None:
                        _bind_root_prefix(req, self.tree_cache)
                    tree_cache = None
                else:
                    tree_cache = self.tree_cache
                req.init_next_round_input(tree_cache)
                # Truncate fill_len to kv_committed_len so checkpoint
                # only sees committed KV (full array includes one uncommitted
                # token because init_next_round_input rebuilt it as full).
                if req.kv.kv_committed_len is not None:
                    req.extend_end = req.kv.kv_committed_len
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        if self.enable_overlap:
            # A finished request can still have one redundant forward in flight.
            # Drain it before a prebuilt request seeds a potentially reused row.
            self.schedule_stream.wait_stream(self.forward_stream)
        return new_batch

    def _start_host_restores(self: Scheduler) -> None:
        """Start the host-to-device copy of every waiting request received into
        host memory. A request joins a batch only once its copy has finished on
        every rank that shares it (`_host_restores_ready`), so the scheduler does
        not wait for a copy on the host; under PP, admission waits for it there."""
        reqs = [
            req for req in self.waiting_queue if req.kv.retraction_backup is not None
        ]
        if not reqs:
            return
        engine = self.tree_cache.cache_controller.l2_transfer_engine
        # With the kernel backend, the copies and their index kernels run on the
        # stream the forwards run on: after the in-flight forward (a copy may
        # write a reused row) and before this iteration's forward, never beside
        # one. The direct backend reads the device indices on the host while
        # preparing a copy, which there would wait for the in-flight forward, so
        # its copies stay on the transfer engine's streams.
        on_forward = get_memory().hicache_io_backend == "kernel"
        # As in run_batch: without overlap or PP, forwards run on the schedule
        # stream.
        forward_stream = (
            self.forward_stream
            if self.enable_overlap or get_parallel().pp_size > 1
            else self.schedule_stream
        )
        if on_forward:
            # They read rows allocated on the schedule stream.
            forward_stream.wait_stream(self.schedule_stream)
        elif self.enable_overlap:
            # A finished request can still have one redundant forward in flight,
            # and a copy may write a reused row: order the copies after it. Fence
            # only the copy stream, as HiCache loads do: index preparation must
            # not wait for the forward (the direct backend reads it on the host).
            engine.host_to_device_stream.wait_stream(self.forward_stream)
        with contextlib.ExitStack() as placement:
            if on_forward:
                placement.enter_context(torch.cuda.stream(forward_stream))
                placement.enter_context(engine.on_current_stream())
            for req in reqs:
                start_restore_kv_cache(req, self.tree_cache)

    def _host_restores_ready(self: Scheduler) -> set[int]:
        """Ids of the waiting requests whose restore copy has finished on every
        rank that shares them. Each rank admits requests on its own and all must
        admit the same ones, so a copy that has finished on only some ranks keeps
        the request waiting on all of them: the attention-TP ranks, then the
        attention-CP ranks, agree through a MIN all-reduce.

        PP stages cannot agree here: a stage selects a microbatch's batch before
        the next stage starts that microbatch, so a collective would deadlock.
        Under PP every pending restore counts as ready, and admission
        (`finish_restore_kv_cache`) waits for the copy on the host, as the
        synchronous restore did, so every stage admits the same requests."""
        pending = [
            req for req in self.waiting_queue if req.kv.restore_done_event is not None
        ]
        if not pending:
            return set()
        if get_parallel().pp_size > 1:
            return {id(req) for req in pending}
        ready = torch.tensor(
            [int(req.kv.restore_done_event.query()) for req in pending],
            dtype=torch.int32,
        )
        for group in (self.attn_tp_cpu_group, self.attn_cp_cpu_group):
            if group is not None and torch.distributed.get_world_size(group) > 1:
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=group
                )
        return {id(req) for req, done in zip(pending, ready.tolist()) if done}

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def process_decode_queue(self: Scheduler):
        if self.enable_decode_hicache:
            self.tree_cache.check_hicache_events()

        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # Resolve held releases every iteration (before the retraction/polling
        # gates below) so their timeouts fire under memory pressure.
        self.disagg_decode_transfer_queue.resolve_deferred_releases()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = get_disagg().disaggregation_decode_polling_interval

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            if get_disagg().disaggregation_decode_host_receive_threshold == 1:
                req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
                self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(transferred_reqs)
            if get_disagg().disaggregation_decode_host_receive_threshold < 1:
                # Give completed host transfers device space before new arrivals.
                req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
                self.disagg_decode_transfer_queue.extend(req_conns)
