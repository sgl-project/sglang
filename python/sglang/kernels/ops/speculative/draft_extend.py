"""Fused metadata construction for EAGLE draft-extend."""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class DraftExtendPrologOutput:
    prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    positions: torch.Tensor
    extend_start_loc: torch.Tensor
    post_extend_seq_lens: torch.Tensor


@triton.jit
def _fused_draft_extend_prolog_kernel(
    seq_lens,
    prefix_lens,
    extend_seq_lens,
    positions,
    extend_start_loc,
    output_seq_lens,
    seq_lens_stride: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    front_offset: tl.constexpr,
    window_size: tl.constexpr,
    WRITE_POSITIONS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request_idx = tl.program_id(0)
    seq_len = tl.load(seq_lens + request_idx * seq_lens_stride)
    prefix_len = tl.maximum(seq_len - front_offset, 0)
    start_loc = request_idx * window_size

    tl.store(prefix_lens + request_idx, prefix_len.to(tl.int32))
    tl.store(extend_seq_lens + request_idx, window_size)
    tl.store(extend_start_loc + request_idx, start_loc)
    tl.store(output_seq_lens + request_idx, seq_len + num_draft_tokens)

    if WRITE_POSITIONS:
        offsets = tl.arange(0, BLOCK_SIZE)
        tl.store(
            positions + start_loc + offsets,
            (prefix_len + offsets).to(tl.int64),
            mask=offsets < window_size,
        )


def fused_draft_extend_prolog(
    seq_lens: torch.Tensor,
    num_draft_tokens: int,
    *,
    front_offset: int = 0,
    positions: Optional[torch.Tensor] = None,
) -> DraftExtendPrologOutput:
    """Build the uniform-width metadata consumed by draft-extend.

    ``positions`` may be supplied by boundary-window widening. In that case the
    kernel preserves it while still producing the remaining metadata.
    """
    assert seq_lens.is_cuda and seq_lens.ndim == 1
    assert seq_lens.dtype in (torch.int32, torch.int64)
    assert num_draft_tokens > 0 and front_offset >= 0

    batch_size = seq_lens.numel()
    window_size = num_draft_tokens + front_offset
    device = seq_lens.device
    prefix_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    extend_seq_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    extend_start_loc = torch.empty((batch_size,), dtype=torch.int32, device=device)
    output_seq_lens = torch.empty((batch_size,), dtype=seq_lens.dtype, device=device)

    write_positions = positions is None
    if write_positions:
        positions = torch.empty(
            (batch_size * window_size,), dtype=torch.int64, device=device
        )
    else:
        assert positions.device == device
        assert positions.dtype == torch.int64
        assert positions.shape == (batch_size * window_size,)
        assert positions.is_contiguous()

    if batch_size > 0:
        _fused_draft_extend_prolog_kernel[(batch_size,)](
            seq_lens,
            prefix_lens,
            extend_seq_lens,
            positions,
            extend_start_loc,
            output_seq_lens,
            seq_lens.stride(0),
            num_draft_tokens,
            front_offset,
            window_size,
            WRITE_POSITIONS=write_positions,
            BLOCK_SIZE=triton.next_power_of_2(window_size),
        )

    return DraftExtendPrologOutput(
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        positions=positions,
        extend_start_loc=extend_start_loc,
        post_extend_seq_lens=output_seq_lens,
    )
