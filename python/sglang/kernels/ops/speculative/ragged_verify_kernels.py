from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl


class PaddedToBucket:
    @classmethod
    def execute(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
    ) -> torch.Tensor:
        impl = cls.triton if verify_lens.is_cuda else cls.torch
        return impl(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
        )

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket_triton(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
        )


def pad_verify_lens_to_bucket(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
) -> torch.Tensor:
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
        "batch's requests"
    )
    device = verify_lens.device
    num_pad_reqs = padded_bs - bs
    padded = verify_lens.to(torch.int32)
    leftover = graph_num_tokens - padded.to(torch.int64).sum()
    if num_pad_reqs > 0:
        base = leftover // num_pad_reqs
        rem = leftover - base * num_pad_reqs
        pad_block = base + (
            torch.arange(num_pad_reqs, device=device, dtype=torch.int64) < rem
        )
        padded = torch.cat([padded, pad_block.to(torch.int32)])
    else:
        padded = padded.clone()
        padded[-1] = (padded[-1].to(torch.int64) + leftover).to(torch.int32)
    return padded


@triton.jit
def _padded_to_bucket_kernel(
    verify_lens_ptr,
    out_ptr,
    bs,
    padded_bs,
    graph_num_tokens,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    valid = idx < padded_bs
    is_real = idx < bs
    vl = tl.load(verify_lens_ptr + idx, mask=is_real, other=0).to(tl.int64)
    leftover = graph_num_tokens - tl.sum(vl)
    num_pad = padded_bs - bs
    num_pad_safe = tl.maximum(num_pad, 1)
    base = leftover // num_pad_safe
    rem = leftover - base * num_pad_safe
    pad_len = base + tl.where((idx - bs) < rem, 1, 0)
    final = tl.where(is_real, vl, pad_len)
    final = final + tl.where((num_pad == 0) & (idx == bs - 1), leftover, 0)
    tl.store(out_ptr + idx, final.to(tl.int32), mask=valid)


def pad_verify_lens_to_bucket_triton(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
) -> torch.Tensor:
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
        "batch's requests"
    )
    device = verify_lens.device
    verify_lens = verify_lens.to(torch.int32).contiguous()
    out = torch.empty(padded_bs, dtype=torch.int32, device=device)
    BLOCK = triton.next_power_of_2(max(padded_bs, 1))
    _padded_to_bucket_kernel[(1,)](
        verify_lens,
        out,
        bs,
        padded_bs,
        graph_num_tokens,
        BLOCK=BLOCK,
    )
    return out


class QoIndptrResult(msgspec.Struct):
    qo_indptr: torch.Tensor
    extend_start_loc: torch.Tensor


class BuildQoIndptr:
    @classmethod
    def execute(cls, *, verify_lens: torch.Tensor) -> QoIndptrResult:
        impl = cls.triton if verify_lens.is_cuda else cls.torch
        return impl(verify_lens=verify_lens)

    @classmethod
    def torch(cls, *, verify_lens: torch.Tensor) -> QoIndptrResult:
        return build_qo_indptr(verify_lens=verify_lens)

    @classmethod
    def triton(cls, *, verify_lens: torch.Tensor) -> QoIndptrResult:
        return build_qo_indptr_triton(verify_lens=verify_lens)


def build_qo_indptr(*, verify_lens: torch.Tensor) -> QoIndptrResult:
    verify_lens = verify_lens.to(torch.int32)
    cumsum = torch.cumsum(verify_lens, dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=verify_lens.device)
    qo_indptr = torch.cat([zero, cumsum])
    extend_start_loc = qo_indptr[:-1].clone()
    return QoIndptrResult(qo_indptr=qo_indptr, extend_start_loc=extend_start_loc)


@triton.jit
def _qo_indptr_kernel(
    verify_lens_ptr,
    qo_indptr_ptr,
    extend_start_loc_ptr,
    bs,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    valid = idx < bs
    vl = tl.load(verify_lens_ptr + idx, mask=valid, other=0).to(tl.int32)
    incl = tl.cumsum(vl, axis=0)
    excl = incl - vl
    tl.store(qo_indptr_ptr, 0)
    tl.store(qo_indptr_ptr + 1 + idx, incl, mask=valid)
    tl.store(extend_start_loc_ptr + idx, excl, mask=valid)


def build_qo_indptr_triton(*, verify_lens: torch.Tensor) -> QoIndptrResult:
    bs = verify_lens.shape[0]
    device = verify_lens.device
    verify_lens = verify_lens.contiguous()
    qo_indptr = torch.empty(bs + 1, dtype=torch.int32, device=device)
    extend_start_loc = torch.empty(bs, dtype=torch.int32, device=device)
    BLOCK = triton.next_power_of_2(max(bs, 1))
    _qo_indptr_kernel[(1,)](
        verify_lens,
        qo_indptr,
        extend_start_loc,
        bs,
        BLOCK=BLOCK,
    )
    return QoIndptrResult(qo_indptr=qo_indptr, extend_start_loc=extend_start_loc)


@triton.jit
def _fill_verify_padding_rows_kernel(
    probs_ptr,
    verify_lens_ptr,
    uniform_ptr,
    width,
    vocab,
    RESTORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = row // width
    verify_len = tl.load(verify_lens_ptr + req).to(tl.int64)
    if row - req * width >= verify_len:
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < vocab
        if RESTORE:
            vals = tl.zeros([BLOCK], dtype=tl.float32) + tl.load(uniform_ptr + row)
        else:
            src = req * width + tl.maximum(verify_len - 1, 0)
            vals = tl.load(probs_ptr + src * vocab + offs, mask=mask)
        tl.store(probs_ptr + row * vocab + offs, vals, mask=mask)


def fill_verify_padding_rows(
    probs: torch.Tensor,
    verify_lens: torch.Tensor,
    uniform: torch.Tensor,
    width: int,
    restore: bool,
) -> None:
    """Requires contiguous ``[bs * width, vocab]`` probs and per-row ``uniform``."""
    vocab = probs.shape[1]
    block = 2048
    _fill_verify_padding_rows_kernel[(probs.shape[0], triton.cdiv(vocab, block))](
        probs,
        verify_lens,
        uniform,
        width,
        vocab,
        RESTORE=restore,
        BLOCK=block,
    )
