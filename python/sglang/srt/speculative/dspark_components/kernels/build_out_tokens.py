from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_BUILD_OUT_TOKENS.get()


class BuildOutTokens:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        verify_num_draft_tokens: int,
        gamma: int,
    ) -> torch.Tensor:
        return build_out_tokens(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=verify_num_draft_tokens,
            gamma=gamma,
        )

    @classmethod
    def triton(
        cls,
        *,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        verify_num_draft_tokens: int,
        gamma: int,
    ) -> torch.Tensor:
        return build_out_tokens_triton(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=verify_num_draft_tokens,
            gamma=gamma,
        )


def build_out_tokens(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    bs = draft_tokens.shape[0]
    out_tokens = torch.empty(
        (bs, verify_num_draft_tokens),
        dtype=torch.int64,
        device=draft_tokens.device,
    )
    out_tokens[:, :gamma].copy_(draft_tokens)
    out_tokens[:, gamma].fill_(0)
    out_tokens.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
    return out_tokens


@triton.jit
def _build_out_tokens_kernel(
    draft_tokens_ptr,
    correct_len_ptr,
    bonus_ptr,
    out_ptr,
    gamma,
    T,
    n_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    b = offs // T
    k = offs % T
    cl = tl.load(correct_len_ptr + b, mask=mask, other=0).to(tl.int32)
    bonus = tl.load(bonus_ptr + b, mask=mask, other=0)
    draft_mask = mask & (k < gamma)
    draft = tl.load(draft_tokens_ptr + b * gamma + k, mask=draft_mask, other=0)
    val = tl.where(k == cl, bonus, tl.where(k < gamma, draft, 0))
    tl.store(out_ptr + offs, val.to(tl.int64), mask=mask)


def build_out_tokens_triton(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    bs = draft_tokens.shape[0]
    T = verify_num_draft_tokens
    device = draft_tokens.device
    draft_tokens = draft_tokens.to(torch.int64).contiguous()
    correct_len_i = correct_len.to(torch.int64).contiguous()
    bonus_i = bonus.to(torch.int64).contiguous()
    out = torch.empty((bs, T), dtype=torch.int64, device=device)
    n_out = bs * T
    BLOCK = 256
    grid = (triton.cdiv(n_out, BLOCK),)
    _build_out_tokens_kernel[grid](
        draft_tokens, correct_len_i, bonus_i, out, gamma, T, n_out, BLOCK=BLOCK
    )
    return out
