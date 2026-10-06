from __future__ import annotations

import random

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["seed"])
def _simulated_bonus_partial_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    part_val_ptr,
    part_idx_ptr,
    seed,
    vocab,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    t = tl.load(temperatures_ptr + row // ROWS).to(tl.float32)
    inv_t = 1.0 / tl.maximum(t, 1e-5)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    u = tl.rand(seed, row * vocab + offs)
    v = tl.where(mask, x * inv_t - tl.log(-tl.log(u)), float("-inf"))
    top = tl.max(v, axis=0)
    idx = tl.min(tl.where(v == top, offs, vocab), axis=0)
    tl.store(part_val_ptr + row * NSPLIT + split, top)
    tl.store(part_idx_ptr + row * NSPLIT + split, idx)


@triton.jit
def _simulated_bonus_select_kernel(
    part_val_ptr,
    part_idx_ptr,
    correct_len_ptr,
    out_ptr,
    vocab,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    req = tl.program_id(0)
    row = req * ROWS + tl.load(correct_len_ptr + req)
    s = tl.arange(0, NSPLIT)
    val = tl.load(part_val_ptr + row * NSPLIT + s)
    idx = tl.load(part_idx_ptr + row * NSPLIT + s)
    top = tl.max(val, axis=0)
    tl.store(
        out_ptr + req, tl.min(tl.where(val == top, idx, vocab), axis=0).to(tl.int64)
    )


def simulated_bonus_sample(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    temperatures: torch.Tensor,
    bs: int,
    rows_per_request: int,
) -> torch.Tensor:
    """Gumbel-max temperature sample of every verify row; returns, per request, the
    sample of the row at ``correct_len``. [bs] int64."""
    assert target_logits.stride(1) == 1
    device = target_logits.device
    vocab = target_logits.shape[1]
    block = 4096
    nsplit = triton.next_power_of_2(triton.cdiv(vocab, block))
    rows = bs * rows_per_request
    part_val = torch.empty((rows, nsplit), dtype=torch.float32, device=device)
    part_idx = torch.empty((rows, nsplit), dtype=torch.int32, device=device)
    _simulated_bonus_partial_kernel[(rows, nsplit)](
        target_logits,
        target_logits.stride(0),
        temperatures.view(-1),
        part_val,
        part_idx,
        random.getrandbits(31),
        vocab,
        ROWS=rows_per_request,
        NSPLIT=nsplit,
        BLOCK=block,
        num_warps=8,
    )
    out = torch.empty((bs,), dtype=torch.int64, device=device)
    _simulated_bonus_select_kernel[(bs,)](
        part_val,
        part_idx,
        correct_len.contiguous(),
        out,
        vocab,
        ROWS=rows_per_request,
        NSPLIT=nsplit,
    )
    return out
