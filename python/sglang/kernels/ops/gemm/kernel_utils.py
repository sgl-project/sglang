import triton
import triton.language as tl


@triton.jit
def _resolve_token_positions(
    sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER: tl.constexpr
):
    """Map logical segment offsets to physical token positions.

    When SORTED_BY_ADAPTER is True, segments are grouped by adapter and
    sorted_token_ids provides the indirection to the original token rows.
    When False, tokens are already contiguous starting at seg_start.
    """
    if SORTED_BY_ADAPTER:
        return tl.load(
            sorted_token_ids + seg_start + s_offset, mask=s_offset < seg_len
        ).to(tl.int64)
    return (seg_start + s_offset).to(tl.int64)
