from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dspark_components.kernels.dispatch import (
    inputs_on_cuda,
)


class BuildBlockSeqLensCausal:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        seq_lens: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_block_seq_lens_causal(
            seq_lens=seq_lens,
            block_size=block_size,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        seq_lens: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_block_seq_lens_causal_triton(
            seq_lens=seq_lens,
            block_size=block_size,
            device=device,
        )


def build_block_seq_lens_causal(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    prefix = seq_lens.to(torch.int32)
    steps = torch.arange(1, block_size + 1, device=device, dtype=torch.int32)
    return (prefix[:, None] + steps[None, :]).reshape(-1)


@triton.jit
def _block_seq_lens_casual_kernel(
    seq_lens_ptr,
    out_ptr,
    block_size,
    n_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    row = offs // block_size
    col = offs % block_size
    prefix = tl.load(seq_lens_ptr + row, mask=mask, other=0)
    tl.store(out_ptr + offs, (prefix + col + 1).to(tl.int32), mask=mask)


def build_block_seq_lens_causal_triton(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    seq_lens = seq_lens.to(device=device, dtype=torch.int64).contiguous()
    n_rows = seq_lens.shape[0]
    n_out = n_rows * block_size
    out = torch.empty(n_out, dtype=torch.int32, device=device)
    BLOCK = 256
    grid = (triton.cdiv(n_out, BLOCK),)
    _block_seq_lens_casual_kernel[grid](seq_lens, out, block_size, n_out, BLOCK=BLOCK)
    return out
