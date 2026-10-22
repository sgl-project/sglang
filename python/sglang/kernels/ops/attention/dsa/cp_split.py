"""Interleave CP q-sequence split kernel for DSA prefill.

Migrated from ``sglang.srt.layers.attention.dsa.utils`` (RFC #29630, Phase 2.5).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_cp_interleave_q_seqs_kernel(
    in_seqs_ptr,
    out_seqs_ptr,
    bs_idx_ptr,
    tokens: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    extra_seq = 0
    bs_idx = 0
    for bs in range(tokens):
        cur_len = tl.load(in_seqs_ptr + bs)
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)
        if cur_seq > 0:
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1
        extra_seq = cur_len - cur_seq * cp_size


def dsa_cp_interleave_q_seqs(
    extend_seqs: torch.Tensor, num_rank_seqs: int, cp_size: int, cp_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's interleaved share of each request's extend length, as
    ``(q_lens, bs_idx)`` over the ``num_rank_seqs`` requests left non-empty."""
    q_lens = torch.empty(
        (num_rank_seqs,), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty((num_rank_seqs,), device=extend_seqs.device, dtype=torch.int32)
    _dsa_cp_interleave_q_seqs_kernel[(1,)](
        extend_seqs, q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return q_lens, bs_idx
