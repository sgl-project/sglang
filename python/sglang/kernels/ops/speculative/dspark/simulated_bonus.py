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


@triton.jit
def _simulated_bonus_row_partial_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    correct_len_ptr,
    correct_len_stride,
    seed_ptr,
    part_val_ptr,
    part_idx_ptr,
    vocab,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    split = tl.program_id(1)
    row = req * ROWS + tl.load(correct_len_ptr + req * correct_len_stride)
    t = tl.load(temperatures_ptr + req).to(tl.float32)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    seed = tl.load(seed_ptr)
    u = tl.rand(seed, req * vocab + offs)
    noisy = x / tl.maximum(t, 1e-5) - tl.log(-tl.log(u))
    v = tl.where(mask, tl.where(t > 0.0, noisy, x), float("-inf"))
    top = tl.max(v, axis=0)
    idx = tl.min(tl.where(v == top, offs, vocab), axis=0)
    tl.store(part_val_ptr + req * NSPLIT + split, top)
    tl.store(part_idx_ptr + req * NSPLIT + split, idx)


@triton.jit
def _simulated_bonus_row_select_kernel(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    vocab,
    NSPLIT: tl.constexpr,
):
    req = tl.program_id(0)
    s = tl.arange(0, NSPLIT)
    val = tl.load(part_val_ptr + req * NSPLIT + s)
    idx = tl.load(part_idx_ptr + req * NSPLIT + s)
    top = tl.max(val, axis=0)
    tl.store(
        out_ptr + req, tl.min(tl.where(val == top, idx, vocab), axis=0).to(tl.int64)
    )


def simulated_bonus_sample_graph(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    temperatures: torch.Tensor,
    seed: torch.Tensor,
    bs: int,
    rows_per_request: int,
) -> torch.Tensor:
    """Graph-safe variant of simulated_bonus_sample: seed ([1] int) and per-request
    temperatures ([bs] float; <= 0 means argmax) are read from device buffers, and
    only the row at ``correct_len`` is sampled. [bs] int64."""
    part_val, part_idx = simulated_bonus_partials_graph(
        target_logits=target_logits,
        correct_len=correct_len,
        temperatures=temperatures,
        seed=seed,
        bs=bs,
        rows_per_request=rows_per_request,
    )
    out = torch.empty((bs,), dtype=torch.int64, device=target_logits.device)
    _simulated_bonus_row_select_kernel[(bs,)](
        part_val, part_idx, out, target_logits.shape[1], NSPLIT=part_val.shape[1]
    )
    return out


def simulated_bonus_partials_graph(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    temperatures: torch.Tensor,
    seed: torch.Tensor,
    bs: int,
    rows_per_request: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """simulated_bonus_sample_graph's per-split (value, index) maxima, [bs, NSPLIT],
    reduced by ``simulated_bonus_select_partials``. ``correct_len`` may be stride-0."""
    assert target_logits.stride(1) == 1
    assert correct_len.dim() == 1 and correct_len.shape[0] == bs
    device = target_logits.device
    vocab = target_logits.shape[1]
    block = 4096
    nsplit = triton.next_power_of_2(triton.cdiv(vocab, block))
    part_val = torch.empty((bs, nsplit), dtype=torch.float32, device=device)
    part_idx = torch.empty((bs, nsplit), dtype=torch.int32, device=device)
    _simulated_bonus_row_partial_kernel[(bs, nsplit)](
        target_logits,
        target_logits.stride(0),
        temperatures,
        correct_len,
        correct_len.stride(0),
        seed,
        part_val,
        part_idx,
        vocab,
        ROWS=rows_per_request,
        NSPLIT=nsplit,
        BLOCK=block,
        num_warps=8,
    )
    return part_val, part_idx


def simulated_bonus_select_partials(
    *, part_val: torch.Tensor, part_idx: torch.Tensor, vocab: int
) -> torch.Tensor:
    """Torch reference of _simulated_bonus_row_select_kernel: the largest value,
    ties to the lowest index. [bs] int64."""
    top = part_val.max(dim=1, keepdim=True).values
    sentinel = torch.full_like(part_idx, vocab)
    return torch.where(part_val == top, part_idx, sentinel).min(dim=1).values.long()


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
