from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dspark_components.kernels.dispatch import (
    inputs_on_cuda,
)


class MixedAcceptSelectResult(msgspec.Struct):
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor


class SelectMixedAccept:
    @classmethod
    def execute(cls, *args, **kwargs) -> MixedAcceptSelectResult:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        greedy_mask: torch.Tensor,
        greedy_len: torch.Tensor,
        greedy_bonus: torch.Tensor,
        greedy_trim: torch.Tensor,
        sampling_len: torch.Tensor,
        sampling_bonus: torch.Tensor,
        sampling_trim: torch.Tensor,
    ) -> MixedAcceptSelectResult:
        return select_mixed_accept(
            greedy_mask=greedy_mask,
            greedy_len=greedy_len,
            greedy_bonus=greedy_bonus,
            greedy_trim=greedy_trim,
            sampling_len=sampling_len,
            sampling_bonus=sampling_bonus,
            sampling_trim=sampling_trim,
        )

    @classmethod
    def triton(
        cls,
        *,
        greedy_mask: torch.Tensor,
        greedy_len: torch.Tensor,
        greedy_bonus: torch.Tensor,
        greedy_trim: torch.Tensor,
        sampling_len: torch.Tensor,
        sampling_bonus: torch.Tensor,
        sampling_trim: torch.Tensor,
    ) -> MixedAcceptSelectResult:
        return select_mixed_accept_triton(
            greedy_mask=greedy_mask,
            greedy_len=greedy_len,
            greedy_bonus=greedy_bonus,
            greedy_trim=greedy_trim,
            sampling_len=sampling_len,
            sampling_bonus=sampling_bonus,
            sampling_trim=sampling_trim,
        )


def select_mixed_accept(
    *,
    greedy_mask: torch.Tensor,
    greedy_len: torch.Tensor,
    greedy_bonus: torch.Tensor,
    greedy_trim: torch.Tensor,
    sampling_len: torch.Tensor,
    sampling_bonus: torch.Tensor,
    sampling_trim: torch.Tensor,
) -> MixedAcceptSelectResult:
    correct_len = torch.where(
        greedy_mask, greedy_len.to(sampling_len.dtype), sampling_len
    )
    bonus = torch.where(greedy_mask, greedy_bonus, sampling_bonus)
    cap_trim_lens = torch.where(
        greedy_mask, greedy_trim.to(sampling_trim.dtype), sampling_trim
    )
    return MixedAcceptSelectResult(
        correct_len=correct_len, bonus=bonus, cap_trim_lens=cap_trim_lens
    )


@triton.jit
def _mixed_accept_select_kernel(
    greedy_mask_ptr,
    greedy_len_ptr,
    greedy_bonus_ptr,
    greedy_trim_ptr,
    sampling_len_ptr,
    sampling_bonus_ptr,
    sampling_trim_ptr,
    correct_len_ptr,
    bonus_ptr,
    cap_trim_ptr,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    is_greedy = tl.load(greedy_mask_ptr + offs, mask=mask, other=0) != 0

    g_len = tl.load(greedy_len_ptr + offs, mask=mask, other=0)
    s_len = tl.load(sampling_len_ptr + offs, mask=mask, other=0)
    tl.store(correct_len_ptr + offs, tl.where(is_greedy, g_len, s_len), mask=mask)

    g_bonus = tl.load(greedy_bonus_ptr + offs, mask=mask, other=0)
    s_bonus = tl.load(sampling_bonus_ptr + offs, mask=mask, other=0)
    tl.store(bonus_ptr + offs, tl.where(is_greedy, g_bonus, s_bonus), mask=mask)

    g_trim = tl.load(greedy_trim_ptr + offs, mask=mask, other=0)
    s_trim = tl.load(sampling_trim_ptr + offs, mask=mask, other=0)
    tl.store(cap_trim_ptr + offs, tl.where(is_greedy, g_trim, s_trim), mask=mask)


def select_mixed_accept_triton(
    *,
    greedy_mask: torch.Tensor,
    greedy_len: torch.Tensor,
    greedy_bonus: torch.Tensor,
    greedy_trim: torch.Tensor,
    sampling_len: torch.Tensor,
    sampling_bonus: torch.Tensor,
    sampling_trim: torch.Tensor,
) -> MixedAcceptSelectResult:
    bs = greedy_mask.shape[0]
    device = greedy_mask.device

    correct_len = torch.empty(bs, dtype=sampling_len.dtype, device=device)
    bonus = torch.empty(bs, dtype=sampling_bonus.dtype, device=device)
    cap_trim_lens = torch.empty(bs, dtype=sampling_trim.dtype, device=device)
    BLOCK = 256
    _mixed_accept_select_kernel[(triton.cdiv(bs, BLOCK),)](
        greedy_mask,
        greedy_len,
        greedy_bonus,
        greedy_trim,
        sampling_len,
        sampling_bonus,
        sampling_trim,
        correct_len,
        bonus,
        cap_trim_lens,
        bs,
        BLOCK=BLOCK,
    )
    return MixedAcceptSelectResult(
        correct_len=correct_len, bonus=bonus, cap_trim_lens=cap_trim_lens
    )
