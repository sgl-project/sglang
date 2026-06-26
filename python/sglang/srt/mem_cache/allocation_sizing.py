from __future__ import annotations

from typing import Optional

from sglang.srt.server_args import ServerArgs, get_global_server_args


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        server_args = get_global_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per
    # decode step (draft chain and verify block share the reservation).

    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.max_speculative_num_draft_tokens
    page_size = server_args.page_size

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # page_size > 1 + topk > 1 (spec v2 tree): worst-case page-aligned tree
        # footprint. Per topk branch needs ceil((last_page_len + num_steps) / page)
        # pages; the partial tail page can be up to page_size - 1, and each branch
        # gets its own (duplicated) copy -- so reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode(server_args)


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation (kv_committed_len +
    get_alloc_reserve_per_decode). The spec v2 page>1 topk>1 holey draft footprint
    can outgrow the default num_draft_tokens headroom (PR #26972).
    """
    # FIXME(lsyin): this is the temporary fix for the context length issue when
    # using speculative decoding
    extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
    if (
        server_args.speculative_algorithm is not None
        and server_args.page_size > 1
        and (server_args.speculative_eagle_topk or 1) > 1
    ):
        extra = max(extra, get_alloc_reserve_per_decode(server_args))
    return extra
