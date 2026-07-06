from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_FINALIZE_ACCEPT.get()


class FinalizeAcceptLensResult(msgspec.Struct):
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    cap_trim_lens: torch.Tensor


class FinalizeAcceptLens:
    @classmethod
    def execute(cls, *args, **kwargs) -> FinalizeAcceptLensResult:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> FinalizeAcceptLensResult:
        return finalize_accept_lens(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> FinalizeAcceptLensResult:
        return finalize_accept_lens_triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )


def finalize_accept_lens(
    *,
    correct_len: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> FinalizeAcceptLensResult:
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    return FinalizeAcceptLensResult(
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
        cap_trim_lens=cap_trim_lens.to(torch.int32),
    )


@triton.jit
def _finalize_accept_lens_kernel(
    correct_len_ptr,
    cap_trim_ptr,
    prefix_lens_ptr,
    commit_lens_ptr,
    new_seq_lens_ptr,
    cap_trim_out_ptr,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    commit = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int32) + 1
    prefix = tl.load(prefix_lens_ptr + offs, mask=mask, other=0)
    trim = tl.load(cap_trim_ptr + offs, mask=mask, other=0).to(tl.int32)
    tl.store(commit_lens_ptr + offs, commit, mask=mask)
    tl.store(new_seq_lens_ptr + offs, prefix + commit, mask=mask)
    tl.store(cap_trim_out_ptr + offs, trim, mask=mask)


def finalize_accept_lens_triton(
    *,
    correct_len: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> FinalizeAcceptLensResult:
    bs = correct_len.shape[0]
    device = correct_len.device

    commit_lens = torch.empty(bs, dtype=torch.int32, device=device)
    new_seq_lens = torch.empty(bs, dtype=prefix_lens.dtype, device=device)
    cap_trim_out = torch.empty(bs, dtype=torch.int32, device=device)
    BLOCK = 256
    _finalize_accept_lens_kernel[(triton.cdiv(bs, BLOCK),)](
        correct_len,
        cap_trim_lens,
        prefix_lens,
        commit_lens,
        new_seq_lens,
        cap_trim_out,
        bs,
        BLOCK=BLOCK,
    )
    return FinalizeAcceptLensResult(
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
        cap_trim_lens=cap_trim_out,
    )
