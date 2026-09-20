import torch
import triton
import triton.language as tl


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix,
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

    # NOTE: This can be slow for large bs
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
