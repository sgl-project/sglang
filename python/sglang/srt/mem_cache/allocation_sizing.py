from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, List

from sglang.srt.environ import envs
from sglang.srt.mem_cache.capacity import tokens_fit
from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


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


def get_spec_decode_reserve() -> int:
    """KV length a speculative decode allocation runs ahead of kv_committed_len:
    twice one step's write, a double buffer for the overlap-mode commit lag.

    Reads the bags: adaptive speculative decoding moves the step count and the
    draft-token bound after publish, and this runs per decode batch.
    """
    spec = get_spec()
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


def spec_decode_alloc_lens(reqs, *, page_size: int):
    """Whole-page speculative decode alloc lens: nxt rounds kv_committed_len +
    get_spec_decode_reserve() up to a page so allocated == recorded (unaligned tails
    leak at ps>1)."""
    reserve = get_spec_decode_reserve()
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
    return cur_kv_lens, nxt_kv_lens, num_needed_tokens


def get_spec_decode_reach(page_size: int) -> int:
    """Upper bound of nxt - kv_committed_len in spec_decode_alloc_lens."""
    return get_spec_decode_reserve() + page_size - 1


@functools.cache
def _spec_decode_headroom_pages(hisparse: bool) -> int:
    pages = envs.SGLANG_OPT_SPEC_DECODE_HEADROOM_PAGES.get()
    if pages <= 0:
        return 0
    if hisparse:
        if envs.SGLANG_OPT_SPEC_DECODE_HEADROOM_PAGES.is_set():
            logger.warning(
                "SGLANG_OPT_SPEC_DECODE_HEADROOM_PAGES is ignored under HiSparse."
            )
        return 0
    return min(pages, 8)


def add_spec_decode_headroom_pages(
    batch: ScheduleBatch,
    nxt_kv_lens: List[int],
    num_needed_tokens: int,
) -> tuple[List[int], int]:
    """On a step that allocates anyway, extend every request to the configured
    pages beyond its page-aligned need in the same allocation."""
    page_size = batch.token_to_kv_pool_allocator.page_size
    if num_needed_tokens == 0 or page_size == 1:
        return nxt_kv_lens, num_needed_tokens
    pages = _spec_decode_headroom_pages(batch.hisparse_coordinator is not None)
    if pages == 0:
        return nxt_kv_lens, num_needed_tokens
    row_cap = batch.req_to_token_pool.req_to_token.shape[1] // page_size * page_size
    reserve = get_spec_decode_reserve()
    lens = list(nxt_kv_lens)
    extra = 0
    for i, req in enumerate(batch.reqs):
        need = req.kv.kv_committed_len + reserve
        target = min(
            row_cap,
            (need + page_size - 1) // page_size * page_size + pages * page_size,
        )
        if target > lens[i]:
            extra += target - lens[i]
            lens[i] = target
    if extra == 0:
        return nxt_kv_lens, num_needed_tokens
    # Paged extend evicts up to needed + bs pages; stay clear of that demand.
    if not tokens_fit(
        batch.token_to_kv_pool_allocator,
        num_needed_tokens + len(lens) * page_size + 4 * extra,
    ):
        return nxt_kv_lens, num_needed_tokens
    return lens, num_needed_tokens + extra


def get_req_to_token_headroom() -> int:
    """req_to_token row columns past the model context length, which admission
    keeps the committed length within: the furthest one decode step writes past it.

    A non-speculative step writes one slot per request, and a paged allocator
    keeps the rest of that page off the row. Derived from the same bag leaves as
    the pools beside the row, so the two agree after a post-publish override.
    """
    if get_spec().speculative_algorithm is None:
        return 1
    return get_spec_decode_reach(get_alloc_page_size())


def get_req_to_token_row_width(context_len: int) -> int:
    """The req_to_token row width for a model context of context_len."""
    return context_len + get_req_to_token_headroom()
