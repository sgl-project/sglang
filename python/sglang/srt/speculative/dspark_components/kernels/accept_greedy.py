from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    CapCorrectLen,
)
from sglang.srt.speculative.dspark_components.kernels.dispatch import (
    inputs_on_cuda,
)


class AcceptGreedy:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy_triton(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )


def accept_greedy(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        row_ids = torch.arange(bs, device=target_predict.device)
        bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


@triton.jit
def _gather_row_bonus_kernel(
    table_ptr,
    idx_ptr,
    out_ptr,
    cols,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int64)
    val = tl.load(table_ptr + offs * cols + idx, mask=mask, other=0)
    tl.store(out_ptr + offs, val.to(tl.int64), mask=mask)


def gather_row_bonus_triton(*, table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bs, cols = table.shape
    table = table.contiguous()
    idx = idx.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=table.device)
    BLOCK = 256
    grid = (triton.cdiv(bs, BLOCK),)
    _gather_row_bonus_kernel[grid](table, idx, out, cols, bs, BLOCK=BLOCK)
    return out


def accept_greedy_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        bonus = gather_row_bonus_triton(table=target_predict, idx=correct_len)
    return correct_len, bonus, cap_trim_lens
