from __future__ import annotations

from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)

# "ahead" is a count of tokens past kv_committed_len; *_lens and row_width are
# positions in the req_to_token row.


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


def get_decode_ahead() -> int:
    """Tokens a decode allocation runs ahead of kv_committed_len, before page
    rounding: one for a non-speculative step, twice one step's write for a
    speculative one (a double buffer for the overlap-mode commit lag).

    Reads the bags because adaptive speculative decoding changes the step count
    after publish.
    """
    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1
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
        write_len = spec_tokens + 1
    elif page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        # A decode step writes max(topk * num_steps, num_draft_tokens).
        write_len = max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        write_len = max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)
    return 2 * write_len


def decode_alloc_targets(reqs, *, page_size: int):
    """Per-request decode alloc targets, rounded up to a whole page (an
    unaligned tail leaks at page_size > 1). Returns (allocated_lens,
    target_lens, num_new_tokens); a non-speculative step allocates only one
    slot, so its demand is the page-aligned allocated..target gap, not
    num_new_tokens."""
    ahead = get_decode_ahead()
    allocated_lens = [0] * len(reqs)
    target_lens = [0] * len(reqs)
    num_new_tokens = 0
    for i, r in enumerate(reqs):
        allocated = r.kv.kv_allocated_len
        target = max(
            allocated,
            (r.kv.kv_committed_len + ahead + page_size - 1) // page_size * page_size,
        )
        allocated_lens[i] = allocated
        target_lens[i] = target
        num_new_tokens += target - allocated
    return allocated_lens, target_lens, num_new_tokens


def get_max_decode_ahead(page_size: int) -> int:
    """Upper bound of target - kv_committed_len in decode_alloc_targets."""
    return get_decode_ahead() + page_size - 1


def get_req_to_token_row_width(context_len: int) -> int:
    """Row width for a model context: admission keeps the committed length
    within context_len, and the columns past it cover the furthest
    decode_alloc_targets reaches. Reads the same bag leaves as the pools beside
    the row, so the two agree after a post-publish override.
    """
    return context_len + get_max_decode_ahead(get_alloc_page_size())
