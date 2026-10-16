from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda

# A row carrying all its mass on one token sends the log-odds to infinity.
MAX_LOG_ODDS = 40.0


class DraftTopLogOdds:
    """logit(q_max) of each draft row's temperature-scaled distribution."""

    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return draft_top_log_odds(*args, **kwargs)
        return draft_top_log_odds_torch(*args, **kwargs)


def draft_top_log_odds_torch(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    """``logits``: [bs * rows_per_request, vocab]; ``temperatures``: [bs]."""
    num_rows = logits.shape[0]
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(-1).to(torch.float32), rows_per_request, dim=0
    )
    assert temp_per_row.numel() == num_rows
    scaled = logits.to(torch.float32) / temp_per_row[:, None]
    row_max = scaled.max(dim=-1, keepdim=True).values
    sum_exp = torch.exp(scaled - row_max).sum(dim=-1)
    # The top token contributes exactly e^0 = 1, so the rest sum to sum_exp - 1
    # and logit(q_max) = log(1 / (sum_exp - 1)).
    complement = torch.clamp(sum_exp - 1.0, min=0.0)
    return torch.clamp(-torch.log(complement), min=-MAX_LOG_ODDS, max=MAX_LOG_ODDS)


@triton.jit
def _draft_top_log_odds_kernel(
    logits_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    rows_per_request,
    logits_row_stride,
    MAX_ODDS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_row_stride

    row_max = -float("inf")
    sum_exp = 0.0
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        new_max = tl.maximum(row_max, tl.max(x, axis=0))
        # Masked (-inf) logits may fill whole blocks; keep exp() off -inf - -inf.
        shift = tl.where(new_max == -float("inf"), 0.0, new_max)
        e = tl.where(vmask, tl.exp(x - shift), 0.0)
        sum_exp = sum_exp * tl.exp(row_max - shift) + tl.sum(e, axis=0)
        row_max = new_max

    complement = tl.maximum(sum_exp - 1.0, 0.0)
    feature = -tl.log(complement)
    feature = tl.minimum(tl.maximum(feature, -MAX_ODDS), MAX_ODDS)
    tl.store(out_ptr + row, feature)


def draft_top_log_odds(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    num_rows, vocab = logits.shape
    bs = num_rows // rows_per_request
    assert bs * rows_per_request == num_rows, (
        f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    )
    assert logits.stride(-1) == 1
    temperatures = temperatures.reshape(bs).to(torch.float32).contiguous()
    out = torch.empty((num_rows,), dtype=torch.float32, device=logits.device)
    if num_rows == 0:
        return out
    _draft_top_log_odds_kernel[(num_rows,)](
        logits,
        temperatures,
        out,
        vocab,
        rows_per_request,
        logits.stride(0),
        MAX_ODDS=MAX_LOG_ODDS,
        BLOCK_V=4096,
    )
    return out
