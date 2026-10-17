import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_entries", "num_stops"])
def _min_new_tokens_kernel(
    logits,
    stop_token_ids,
    stop_token_penalties,
    len_output_tokens,
    min_new_tokens,
    num_entries,
    num_stops,
    logits_stride_row: tl.constexpr,
    logits_stride_token: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    entry = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = entry < num_entries
    row = entry // num_stops
    length = tl.load(len_output_tokens + row, valid, other=0)
    minimum = tl.load(min_new_tokens + row, valid, other=0)
    penalty = tl.load(stop_token_penalties + entry, valid, other=0)
    update = valid & (length < minimum) & (penalty != 0)
    token = tl.load(stop_token_ids + entry, update, other=0)
    ptr = logits + row.to(tl.int64) * logits_stride_row + token * logits_stride_token
    value = tl.load(ptr, update, other=0)
    # Stop IDs are unique per row. Ignore zero-valued padding so a padded
    # token 0 never races with an actual stop token 0 in the same row.
    tl.store(ptr, value + penalty, update)


def apply_min_new_tokens_penalties(
    logits: torch.Tensor,
    stop_token_ids: torch.Tensor,
    stop_token_penalties: torch.Tensor,
    len_output_tokens: torch.Tensor,
    min_new_tokens: torch.Tensor,
) -> None:
    """Apply compact penalties; non-padding stop IDs must be unique and in range."""
    num_entries = stop_token_ids.numel()
    if num_entries == 0:
        return
    _min_new_tokens_kernel[(triton.cdiv(num_entries, 256),)](
        logits,
        stop_token_ids,
        stop_token_penalties,
        len_output_tokens,
        min_new_tokens,
        num_entries,
        stop_token_ids.shape[1],
        logits.stride(0),
        logits.stride(1),
        BLOCK_SIZE=256,
    )
