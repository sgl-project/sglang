from __future__ import annotations

from typing import Optional

from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)


def get_mamba_tracking_slots(
    *, extra_buffer: bool, overlap: bool, lazy: bool = False
) -> int:
    """Tracking slots allocated initially; lazy mode defers the second slot."""
    if not extra_buffer:
        return 0
    if lazy:
        return 1
    return 2 if overlap else 1


def get_alloc_page_size() -> int:
    # Mirrors _build_token_to_kv_pool_allocator's DCP branch; the platform
    # allocators that skip it page smaller, so this is an upper bound for them.
    return get_schedule().page_size * get_parallel().attn_dcp_size


def get_alloc_len_per_decode() -> int:
    """KV length one request may allocate in a single decode step.

    Reads the bags: adaptive speculative decoding moves the step count and the
    draft-token bound after publish, and this runs per decode batch.
    """
    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = spec.speculative_num_steps or 1
    spec_topk = spec.speculative_eagle_topk or 1
    spec_tokens = max_speculative_num_draft_tokens()
    page_size = get_alloc_page_size()

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if spec_algo.is_uno():
        if spec_tokens is None:
            raise RuntimeError("UNO requires speculative_num_draft_tokens")
        # UNO retains an additional clean-root position beside Q/F draft slots.
        return spec_tokens + 1
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode() -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode()


def page_aligned_decode_alloc_lens(
    reqs,
    *,
    reserve: int,
    page_size: int,
    free_tokens: Optional[int] = None,
    row_len: Optional[int] = None,
):
    """Whole-page decode alloc lens: nxt rounds committed up to page so allocated
    == recorded (unaligned tails leak at ps>1).

    With ``free_tokens`` (the allocator's free size) and ``row_len`` (the
    req_to_token row width), a request that is short of ``reserve`` is topped up
    to twice ``reserve`` when the pool keeps ``reserve`` free per request after
    the allocation. Its next steps then allocate nothing until the committed
    length catches up, instead of a few slots every step.
    """
    cur_kv_lens = [0] * len(reqs)
    nxt_kv_lens = [0] * len(reqs)
    num_needed_tokens = 0
    for i, r in enumerate(reqs):
        cur = r.kv.kv_allocated_len
        nxt = max(
            cur,
            (r.kv.kv_committed_len + reserve + page_size - 1) // page_size * page_size,
        )
        cur_kv_lens[i] = cur
        nxt_kv_lens[i] = nxt
        num_needed_tokens += nxt - cur
    if num_needed_tokens == 0 or free_tokens is None or row_len is None:
        return cur_kv_lens, nxt_kv_lens, num_needed_tokens

    row_end = row_len // page_size * page_size
    topped = list(nxt_kv_lens)
    for i, r in enumerate(reqs):
        if nxt_kv_lens[i] > cur_kv_lens[i]:
            chunk_end = r.kv.kv_committed_len + 2 * reserve
            chunk_end = (chunk_end + page_size - 1) // page_size * page_size
            topped[i] = max(nxt_kv_lens[i], min(chunk_end, row_end))
    num_topped_tokens = num_needed_tokens + sum(topped) - sum(nxt_kv_lens)
    if free_tokens - num_topped_tokens >= reserve * len(reqs):
        return cur_kv_lens, topped, num_topped_tokens
    return cur_kv_lens, nxt_kv_lens, num_needed_tokens


def get_req_to_token_extra_context_len() -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation; the spec v2 page>1 topk>1 holey
    draft footprint can outgrow the default num_draft_tokens headroom. The row
    headroom and the pools it sits next to derive from the same bag leaves, so
    they cannot disagree after a post-publish override.
    """
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    extra = 4 + (max_speculative_num_draft_tokens() or 0)
    page_size = get_alloc_page_size()
    spec_algorithm = get_spec().speculative_algorithm
    if spec_algorithm is not None:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        spec_algo = SpeculativeAlgorithm.from_string(spec_algorithm)
        if page_size > 1 or spec_algo.is_uno():
            # UNO's double-buffer reserve applies at every page size. Larger
            # pages may additionally round the allocation up by page_size - 1.
            extra = max(extra, get_alloc_reserve_per_decode() + page_size - 1)
        extra = max(extra, spec_algo.req_to_token_extra_len(page_size))
    return extra
