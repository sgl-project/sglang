import torch
import triton
import triton.language as tl


def compute_position_triton(
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_sum,
    *,
    extend_start_loc: torch.Tensor | None = None,
):
    """Compute positions, optionally reusing read-only request offsets.

    Supplied offsets are exclusive query starts (without the terminal offset).
    Their values belong to the caller's layout; this kernel never rewrites them.
    """
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size
    reuse_start_loc = extend_start_loc is not None
    if reuse_start_loc and (
        extend_start_loc.shape != (batch_size,)
        or extend_start_loc.device != extend_seq_lens.device
        or extend_start_loc.dtype not in (torch.int32, torch.int64)
        or not extend_start_loc.is_contiguous()
    ):
        raise ValueError(
            "Request offsets must be contiguous integer [batch_size] on the input device"
        )

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    if not reuse_start_loc:
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
        reuse_start_loc,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
    reuse_start_loc: tl.constexpr = False,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    if reuse_start_loc:
        cumsum_start = tl.load(extend_start_loc + pid).to(tl.int64)
    else:
        # Callers without a prepared layout retain the original prefix scan.
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
    if not reuse_start_loc:
        tl.store(extend_start_loc + pid, cumsum_start)
