import torch
import triton
import triton.language as tl

# Below this batch size the original single-program kernel already saturates the
# SMs; the striped kernel (fewer, larger blocks) only helps once bs is large
# enough that the O(bs^2) serial prefix-sum dominates. Tuned on H200; the
# crossover is lower on few-core accelerators (e.g. Ascend NPU crosses ~bs 50).
_STRIPE_MIN_BS = 1024


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`.

    For large batch sizes the per-request serial prefix-sum in
    ``compute_position_kernel`` is O(bs^2); above ``_STRIPE_MIN_BS`` we dispatch
    to ``compute_position_striped_kernel``, which caps the aggregate prefix-sum
    work at O(min(bs, 64)^2) by fanning the batch across <= 64 stripes. Output
    is byte-identical for both paths.
    """
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    if batch_size < _STRIPE_MIN_BS:
        compute_position_kernel[(batch_size,)](
            positions,
            extend_start_loc,
            extend_prefix_lens,
            extend_seq_lens,
            has_prefix,
        )
        return positions, extend_start_loc

    rows_per_stripe = 16
    num_stripes = triton.cdiv(batch_size, rows_per_stripe)
    if num_stripes > 64:
        num_stripes = 64
        rows_per_stripe = triton.cdiv(batch_size, num_stripes)

    compute_position_striped_kernel[(num_stripes,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        batch_size,
        has_prefix,
        512,  # BLOCK_TOKENS
        rows_per_stripe,
    )
    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs — see compute_position_striped_kernel.
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


@triton.jit
def compute_position_striped_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    batch_size,
    has_prefix: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    ROWS_PER_STRIPE: tl.constexpr,
):
    """Large-bs path: <= 64 stripes, each fills ROWS_PER_STRIPE consecutive rows.

    Each stripe seeds its running offset ONCE (exclusive prefix-sum up to its
    first row), then fills its rows sequentially. The serial prefix chain is
    bounded by (num_stripes - 1) * ROWS_PER_STRIPE, but with <= 64 stripes the
    aggregate prefix-sum work is O(min(bs, 64)^2) instead of the O(bs^2) of the
    one-program-per-request kernel, and the grid shrinks from bs to <= 64 blocks.
    """
    stripe = tl.program_id(0)
    row_begin = stripe * ROWS_PER_STRIPE
    row_end = tl.minimum(row_begin + ROWS_PER_STRIPE, batch_size)

    cumsum_start = tl.cast(0, tl.int64)
    for i in range(row_begin):
        cumsum_start += tl.load(extend_seq_lens + i)

    offsets = tl.arange(0, BLOCK_TOKENS)
    for row in range(row_begin, row_end):
        prefix_len = tl.load(extend_prefix_lens + row) if has_prefix else 0
        seq_len = tl.load(extend_seq_lens + row)
        tl.store(extend_start_loc + row, cumsum_start)
        num_loop = tl.cdiv(seq_len, BLOCK_TOKENS)
        for tile in range(num_loop):
            token_offsets = offsets + tile * BLOCK_TOKENS
            tl.store(
                positions + cumsum_start + token_offsets,
                prefix_len + token_offsets,
                mask=token_offsets < seq_len,
            )
        cumsum_start += seq_len
