"""Stateless Gumbel-max scores for the simulated-acceptance draft benchmark."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Domain separate this Philox stream from draft and verify sampling.
_KEY = 0x5BD1E995


@triton.jit
def _draft_scores_kernel(
    X,
    Y,
    STEP,
    temperature,
    rows_start,
    cols_start,
    cols,
    stride_row,
    stride_col,
    KEY: tl.constexpr,
    STEP_IS_TENSOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if STEP_IS_TENSOR:
        step = tl.load(STEP)
    else:
        step = tl.cast(STEP, tl.int64)
    zeros = tl.full((BLOCK,), 0, tl.uint32)
    word, _, _, _ = tl.philox(
        KEY,
        (cols_start + col).to(tl.uint32),
        zeros + (rows_start + row).to(tl.uint32),
        zeros + step.to(tl.uint32),
        zeros + (step >> 32).to(tl.uint32),
    )
    # Small words win the Gumbel-max race. log1p preserves their precision.
    # The largest words round to 1 in fp32; keep this endpoint finite too.
    u = tl.minimum((word.to(tl.float32) + 0.5) * (2.0**-32), 1.0 - 2.0**-24)
    noise = -tl.log(-libdevice.log1p(-u))
    logits = tl.load(X + row * stride_row + col * stride_col, col < cols, other=0)
    # Scale both terms by min(T, 1). The argmax is invariant and logits stay
    # finite even for temperatures close to float32's smallest normal value.
    score = logits.to(tl.float32) / tl.maximum(temperature, 1.0) + noise * tl.minimum(
        temperature, 1.0
    )
    tl.store(Y + row * cols + col, score, col < cols)


def draft_scores(
    logits: torch.Tensor,
    temperature: float,
    step: torch.Tensor | int,
    row_start: int,
    col_start: int,
) -> torch.Tensor:
    rows, cols = logits.shape
    if logits.is_cuda:
        scores = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        if scores.numel():
            block = 256
            _draft_scores_kernel[(rows, triton.cdiv(cols, block))](
                logits,
                scores,
                step,
                temperature,
                row_start,
                col_start,
                cols,
                *logits.stride(),
                _KEY,
                isinstance(step, torch.Tensor),
                block,
            )
        return scores

    from sglang.kernels.ops.speculative.dspark.markov_sample import _philox

    row = torch.arange(row_start, row_start + rows, device=logits.device)
    col = torch.arange(col_start, col_start + cols, device=logits.device)
    word, _, _, _ = _philox(
        _KEY, col[None, :], row[:, None], step & 0xFFFFFFFF, step >> 32
    )
    u = ((word.float() + 0.5) * 2.0**-32).clamp_max_(1.0 - 2.0**-24)
    noise = -(-torch.log1p(-u)).log()
    return logits.float() / max(temperature, 1.0) + noise * min(temperature, 1.0)
