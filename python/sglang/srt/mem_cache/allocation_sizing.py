from __future__ import annotations

from typing import Optional

from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs


def get_alloc_len_per_decode(
    server_args: ServerArgs, *, max_draft_tokens: Optional[int] = None
) -> int:
    """``max_draft_tokens`` lets a caller that already resolved the draft-token
    bound (the KV-cache configurator reads it off the bags) size with that same
    value; the default is the handed instance's own member, never the global."""
    if server_args.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = (
        max_draft_tokens
        if max_draft_tokens is not None
        else server_args.max_speculative_num_draft_tokens
    )
    page_size = server_args.page_size

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
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


def get_alloc_reserve_per_decode(
    server_args: Optional[ServerArgs] = None,
    *,
    max_draft_tokens: Optional[int] = None,
) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.

    Callers on a request path have no config in hand, so this is the module's
    single "which config" decision point: everything below it takes the
    instance explicitly.
    """
    if server_args is None:
        server_args = get_server_args()
    return 2 * get_alloc_len_per_decode(server_args, max_draft_tokens=max_draft_tokens)


def get_req_to_token_extra_context_len(
    server_args: ServerArgs, *, max_draft_tokens: Optional[int] = None
) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation; the spec v2 page>1 topk>1 holey
    draft footprint can outgrow the default num_draft_tokens headroom.

    ``max_draft_tokens`` keeps this row headroom and the caller's other
    draft-token-sized buffers on ONE resolved value: the KV-cache configurator
    passes the bag-derived bound it also hands the pools, so the two cannot
    disagree after a post-publish override. The default stays the handed
    instance's member for callers sizing against a specific config object.
    """
    if max_draft_tokens is None:
        max_draft_tokens = server_args.max_speculative_num_draft_tokens
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    extra = 4 + (max_draft_tokens or 0)
    if server_args.speculative_algorithm is not None and server_args.page_size > 1:
        # kv_allocated_len is page-aligned (eagle_prepare_for_decode), so near
        # the context limit the aligned reserve can overshoot by page_size - 1;
        # without the headroom the row write silently lands in the neighbor row.
        extra = max(
            extra,
            get_alloc_reserve_per_decode(server_args, max_draft_tokens=max_draft_tokens)
            + server_args.page_size
            - 1,
        )
    return extra
