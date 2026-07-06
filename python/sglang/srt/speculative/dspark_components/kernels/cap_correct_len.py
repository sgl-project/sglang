from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_CAP_CORRECT_LEN.get()


class CapCorrectLen:
    @classmethod
    def execute(cls, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        correct_len: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cap_correct_len(
            correct_len=correct_len,
            verify_lens=verify_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        correct_len: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cap_correct_len_triton(
            correct_len=correct_len,
            verify_lens=verify_lens,
        )


def cap_correct_len(
    *,
    correct_len: torch.Tensor,
    verify_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ell_r = (verify_lens.to(device=correct_len.device) - 1).to(correct_len.dtype)
    capped = torch.minimum(correct_len, ell_r)
    cap_trim_lens = correct_len - capped
    return capped, cap_trim_lens


@triton.jit
def _cap_correct_len_kernel(
    correct_len_ptr,
    verify_lens_ptr,
    capped_ptr,
    trim_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
    vl = tl.load(verify_lens_ptr + offs, mask=mask, other=0).to(tl.int64)
    ell = vl - 1
    capped = tl.minimum(cl, ell)
    trim = cl - capped
    tl.store(capped_ptr + offs, capped, mask=mask)
    tl.store(trim_ptr + offs, trim, mask=mask)


def cap_correct_len_triton(
    *,
    correct_len: torch.Tensor,
    verify_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = correct_len.device
    correct_len = correct_len.contiguous()
    verify_lens = verify_lens.to(device=device).contiguous()
    n = correct_len.shape[0]
    capped = torch.empty_like(correct_len)
    trim = torch.empty_like(correct_len)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _cap_correct_len_kernel[grid](
        correct_len, verify_lens, capped, trim, n, BLOCK=BLOCK
    )
    return capped, trim
