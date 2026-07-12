import torch
import triton
import triton.language as tl


@triton.jit
def _vocab_parallel_embedding_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    # The scalar params are tl.constexpr on purpose: it lets the compiler fold
    # the vocab-window comparisons into a single range check (measured ~5%
    # faster at large token counts), and each embedding layer has one fixed
    # (hidden_dim, stride, shard-window) tuple, so the specialization costs one
    # compile per layer at warmup.
    hidden_dim: tl.constexpr,
    weight_stride0: tl.constexpr,
    org_vocab_start_index: tl.constexpr,
    org_vocab_end_index: tl.constexpr,
    num_org_vocab_padding: tl.constexpr,
    added_vocab_start_index: tl.constexpr,
    added_vocab_end_index: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden_dim

    token = tl.load(input_ptr + row).to(tl.int64)
    org_vocab_mask = (token >= org_vocab_start_index) & (token < org_vocab_end_index)
    added_vocab_mask = (token >= added_vocab_start_index) & (
        token < added_vocab_end_index
    )
    valid = org_vocab_mask | added_vocab_mask

    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    local_id = tl.where(
        org_vocab_mask, token - org_vocab_start_index, token - added_offset
    )
    vals = tl.load(
        weight_ptr + local_id * weight_stride0 + cols,
        mask=col_mask & valid,
        other=0.0,
    )
    tl.store(out_ptr + row * hidden_dim + cols, vals, mask=col_mask)


def vocab_parallel_embedding(
    input_: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    assert input_.is_cuda
    assert input_.is_contiguous()
    assert input_.dtype in (torch.int32, torch.int64)
    assert weight.is_cuda
    assert weight.ndim == 2
    assert weight.stride(1) == 1

    hidden_dim = weight.shape[1]
    output = torch.empty(
        (*input_.shape, hidden_dim), dtype=weight.dtype, device=weight.device
    )
    n_tokens = input_.numel()
    if n_tokens == 0:
        return output

    block_h = min(1024, triton.next_power_of_2(hidden_dim))
    grid = (n_tokens, triton.cdiv(hidden_dim, block_h))
    _vocab_parallel_embedding_kernel[grid](
        input_,
        weight,
        output,
        hidden_dim,
        weight.stride(0),
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return output
