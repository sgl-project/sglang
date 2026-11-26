import torch
import triton
import triton.language as tl


@triton.jit
def _mean_pool_kernel(
    hidden,
    ends,
    output,
    hidden_size: tl.constexpr,
    row_stride: tl.constexpr,
    col_stride: tl.constexpr,
    num_splits: tl.constexpr,
    block_tokens: tl.constexpr,
    block_dim: tl.constexpr,
):
    request = tl.program_id(0)
    split = tl.program_id(1)
    dims = tl.program_id(2) * block_dim + tl.arange(0, block_dim)
    start = tl.load(ends + request - 1, mask=request > 0, other=0)
    end = tl.load(ends + request)
    length = end - start
    lo = start + length * split // num_splits
    hi = start + length * (split + 1) // num_splits

    acc = tl.full((block_tokens, block_dim), 0, tl.float32)
    for offset in range(lo, hi, block_tokens):
        tokens = offset + tl.arange(0, block_tokens)
        values = tl.load(
            hidden + tokens[:, None] * row_stride + dims[None, :] * col_stride,
            mask=(tokens[:, None] < hi) & (dims[None, :] < hidden_size),
            other=0,
        )
        acc += values.to(tl.float32)
    result = tl.sum(acc, axis=0)
    if num_splits == 1:
        result /= length
    tl.store(
        output + (request * num_splits + split) * hidden_size + dims,
        result,
        mask=dims < hidden_size,
    )


def mean_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Pool nonempty packed CUDA tensors using FP32 accumulation.

    lengths holds one device-side length per sequence. Padding past the
    last boundary is ignored and a zero-length sequence produces NaNs.
    """
    tokens, hidden_size = hidden.shape
    batch_size = lengths.numel()
    # Split long sequences across blocks. Bound the temporary independently
    # of the longest request so a short request mixed with it stays cheap.
    num_splits = min(32, triton.next_power_of_2(triton.cdiv(tokens, batch_size * 32)))
    dtype = hidden.dtype if num_splits == 1 else torch.float32
    partials = torch.empty(
        (batch_size, num_splits, hidden_size), device=hidden.device, dtype=dtype
    )
    ends = torch.cumsum(lengths, dim=0)
    _mean_pool_kernel[(batch_size, num_splits, triton.cdiv(hidden_size, 64))](
        hidden, ends, partials, hidden_size, *hidden.stride(), num_splits, 16, 64
    )
    if num_splits == 1:
        return partials[:, 0]
    return (partials.sum(dim=1) / lengths[:, None]).to(hidden.dtype)
