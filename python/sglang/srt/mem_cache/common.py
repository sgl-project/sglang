from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_evict_dsv4_state_on_swa,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    CacheFinishParams,
    CacheUnfinishParams,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.owned_kv import free_swa_out_of_window_slots
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def evict_swa_out_of_window_for_unfinished(
    req: Req,
    tree_cache: BasePrefixCache,
    chunked: bool,
) -> Optional[int]:
    from sglang.srt.environ import envs

    if not envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
        return None
    if not tree_cache.supports_swa():
        return None

    pre_len = tree_cache.unfinished_swa_evict_pre_len(req, chunked=chunked)
    if pre_len is None:
        return None
    if req.kv is None:
        return None

    free_swa_out_of_window_slots(
        req_pool_idx=req.req_pool_idx,
        kv=req.kv,
        owned_start=req.cache.cache_protected_len,
        pre_len=pre_len,
        sliding_window_size=tree_cache.sliding_window_size,
        page_size=tree_cache.page_size,
        req_to_token_pool=tree_cache.req_to_token_pool,
        token_to_kv_pool_allocator=tree_cache.token_to_kv_pool_allocator,
        on_swa_evicted=lambda watermark: maybe_evict_dsv4_state_on_swa(
            tree_cache.token_to_kv_pool_allocator,
            tree_cache.req_to_token_pool,
            req,
            watermark,
        ),
    )
    return req.kv.swa_evicted_seqlen


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    chunked = kwargs.get("chunked", False)
    evict_swa_out_of_window_for_unfinished(req, tree_cache, chunked=chunked)
    harvest_and_cache_unfinished_req(req, tree_cache, chunked=chunked)


def harvest_and_cache_unfinished_req(
    req: Req, tree_cache: BasePrefixCache, chunked: bool = False
) -> None:
    token_ids = req.get_fill_ids()
    kv_indices = tree_cache.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]
    unfinish_params = CacheUnfinishParams(
        token_ids=token_ids,
        extra_key=req.extra_key,
        kv_indices=kv_indices,
        req_pool_idx=req.req_pool_idx,
        prev_prefix_len=req.cache.cache_protected_len,
        prefix_indices_len=len(req.prefix_indices),
        swa_evicted_seqlen=req.kv.swa_evicted_seqlen if req.kv is not None else 0,
        priority=getattr(req, "priority", 0) or 0,
        chunked=chunked,
        last_node=req.cache.last_node,
        swa_uuid_for_lock=req.cache.swa_uuid_for_lock,
        swa_prefix_lock_released=req.cache.swa_prefix_lock_released,
        req=req,
    )
    unfinish_result = tree_cache.cache_unfinished_req(unfinish_params)

    if unfinish_result is None:
        return
    if unfinish_result.prefix_indices is not None:
        req.prefix_indices = unfinish_result.prefix_indices
    if unfinish_result.cache_protected_len is not None:
        req.cache.cache_protected_len = unfinish_result.cache_protected_len
    if unfinish_result.lock_handover:
        req.cache.last_node = unfinish_result.last_node
        req.cache.swa_uuid_for_lock = unfinish_result.swa_uuid_for_lock
        if unfinish_result.swa_prefix_lock_released is not None:
            req.cache.swa_prefix_lock_released = (
                unfinish_result.swa_prefix_lock_released
            )


def harvest_and_finish_req(
    req: Req, tree_cache: BasePrefixCache, is_insert: bool = True
) -> None:
    kv_committed_len = req.effective_kv_committed_len()
    owned_start = req.cache.cache_protected_len
    kv_indices = tree_cache.req_to_token_pool.req_to_token[
        req.req_pool_idx, :kv_committed_len
    ]
    finish_params = CacheFinishParams(
        token_ids=(req.origin_input_ids + req.output_ids)[:kv_committed_len],
        extra_key=req.extra_key,
        kv_indices=kv_indices,
        kv_committed_len=kv_committed_len,
        prev_prefix_len=owned_start,
        prefix_indices_len=len(req.prefix_indices),
        swa_evicted_seqlen=req.kv.swa_evicted_seqlen if req.kv is not None else 0,
        priority=getattr(req, "priority", 0) or 0,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        last_node=req.cache.last_node,
        swa_uuid_for_lock=req.cache.swa_uuid_for_lock,
        swa_prefix_lock_released=req.cache.swa_prefix_lock_released,
        rid=req.rid,
        req=req,
    )
    finish_result = tree_cache.cache_finished_req(finish_params)

    if finish_result is not None and req.kv is not None:
        if finish_result.prefix_len is not None:
            tree_cache.token_to_kv_pool_allocator.free(
                kv_indices[owned_start : finish_result.prefix_len]
            )
        if finish_result.key_len is not None:
            tree_cache.token_to_kv_pool_allocator.free(
                kv_indices[finish_result.key_len :]
            )


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba.mamba_pool_idx = None
        return

    kv_committed_len = req.effective_kv_committed_len()

    harvest_and_finish_req(req, tree_cache, is_insert=is_insert)

    # StreamingSession.cache_finished_req handles speculative tail trim
    # internally, then sets req_pool_idx = None.
    if req.kv is None:
        return

    start_p, end_p = kv_committed_len, req.kv.kv_allocated_len

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv = None
