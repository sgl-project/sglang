"""The fp8 index-query operand of DeepGEMM's Hopper MQA logits.

The fake-fp4 index query ``indexer.queries`` returns is an e2m1 value times a
per-32-block power-of-two scale. One fp8 row carries one scale, so rescale onto
a power of two and fold that factor into the (fp32) head weights: every e2m1
value has at most three significant bits and stays exact under a power-of-two
rescale, modulo blocks more than eight binades below the row's largest.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_index_q_fp8_kernel(
    Q,
    W,
    Q_out,
    W_out,
    n_rows,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows < n_rows
    offs = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + rows[:, None] * HEAD_DIM + offs[None, :], valid[:, None], 0.0).to(
        tl.float32
    )
    amax = tl.max(tl.abs(q), 1)
    # ceil to a power of two: e2m1 magnitudes reach 6, and the floor keeps an
    # all-zero row finite.
    unit = tl.maximum(amax * (1.0 / 6.0), 1.0e-4)
    bits = unit.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    tl.store(
        Q_out + rows[:, None] * HEAD_DIM + offs[None, :],
        (q / scale[:, None]).to(tl.float8e4nv),
        valid[:, None],
    )
    tl.store(W_out + rows, tl.load(W + rows, valid, 0.0) * scale, valid)


def quantize_index_q_fp8(
    q: torch.Tensor, weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp8 queries ``[rows, heads, 128]`` and the fp32 head weights ``[rows, heads]``
    scaled by each row-head's quantization factor, so that
    ``<q, k> == scale * <q_fp8, k>``."""
    rows, heads, head_dim = q.shape
    assert head_dim == 128
    q = q.contiguous()
    q_fp8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    weights = weights.float().contiguous()
    weights_out = torch.empty_like(weights)
    n_rows = rows * heads
    if n_rows:
        block_m = 8
        _quantize_index_q_fp8_kernel[(triton.cdiv(n_rows, block_m),)](
            q,
            weights,
            q_fp8,
            weights_out,
            n_rows,
            BLOCK_M=block_m,
            HEAD_DIM=head_dim,
            num_warps=4,
        )
    return q_fp8, weights_out
