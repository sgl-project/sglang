from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


def resolve_kv_bookkeeping_page(allocator: BaseTokenToKVPoolAllocator) -> int:
    if allocator.uses_legacy_real_length_alloc:
        return 1
    return allocator.page_size


def get_alloc_page_size_upper_bound(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        server_args = get_server_args()
    return server_args.page_size * server_args.dcp_size


def get_alloc_len_per_decode(
    server_args: Optional[ServerArgs] = None, *, page_size: Optional[int] = None
) -> int:
    if server_args is None:
        server_args = get_server_args()
    if page_size is None:
        page_size = server_args.page_size

    if server_args.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.max_speculative_num_draft_tokens or 0

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
    server_args: Optional[ServerArgs] = None, *, page_size: Optional[int] = None
) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode(server_args, page_size=page_size)


def get_candidate_alloc_page_sizes(server_args: ServerArgs) -> Tuple[int, ...]:
    return (server_args.page_size, get_alloc_page_size_upper_bound(server_args))


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation, which every speculative family can
    drive past the default num_draft_tokens headroom.
    """
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
    if server_args.speculative_algorithm is not None:
        extra = max(
            extra,
            *(
                get_alloc_reserve_per_decode(server_args, page_size=page_size)
                for page_size in get_candidate_alloc_page_sizes(server_args)
            ),
        )
    return extra


def get_req_to_token_row_width(
    *, server_args: ServerArgs, model_config: ModelConfig
) -> int:
    extra = get_req_to_token_extra_context_len(server_args)
    return ceil_align(
        model_config.context_len + extra,
        get_alloc_page_size_upper_bound(server_args),
    )


def assert_alloc_within_row_width(*, max_alloc_len: int, row_width: int) -> None:
    assert max_alloc_len <= row_width, (
        f"page-aligned KV allocation ({max_alloc_len}) exceeds req_to_token row "
        f"width ({row_width}); widen the row via get_req_to_token_row_width."
    )
