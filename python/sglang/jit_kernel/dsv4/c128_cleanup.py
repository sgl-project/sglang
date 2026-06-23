import torch
import triton
import triton.language as tl


@triton.jit
def _clear_unaccepted_c128_draft_states_kernel(
    state,
    req_pool_indices,
    seq_lens,
    accept_lens,
    ring_size: tl.constexpr,
    half: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bid = tl.program_id(0)
    draft_offset = tl.program_id(1)
    block_id = tl.program_id(2)

    accept_len = tl.load(accept_lens + bid)
    if draft_offset < accept_len:
        return

    req_pool_idx = tl.load(req_pool_indices + bid).to(tl.int64)
    seq_len = tl.load(seq_lens + bid).to(tl.int64)
    slot = (seq_len + draft_offset) % ring_size
    row = req_pool_idx * ring_size + slot

    offsets = block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < half
    row_base = row * (half * 2)
    tl.store(state + row_base + offsets, 0.0, mask=mask)
    tl.store(state + row_base + half + offsets, float("-inf"), mask=mask)


def clear_unaccepted_c128_draft_states(
    state: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    ring_size: int,
    num_draft_tokens: int,
) -> None:
    half = state.shape[-1] // 2
    _clear_unaccepted_c128_draft_states_kernel[
        (req_pool_indices.numel(), num_draft_tokens, triton.cdiv(half, 256))
    ](
        state,
        req_pool_indices,
        seq_lens,
        accept_lens,
        ring_size,
        half,
        num_draft_tokens,
        BLOCK_D=256,
    )
