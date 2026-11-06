from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl


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


class PaddedLayout(msgspec.Struct):
    verify_lens: torch.Tensor
    qo_indptr: torch.Tensor
    extend_start_loc: torch.Tensor


class PadVerifyLayout:
    @classmethod
    def execute(cls, **kwargs) -> PaddedLayout:
        impl = cls.triton if kwargs["verify_lens"].is_cuda else cls.torch
        return impl(**kwargs)

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        padded_bs: int,
        cap: Optional[int] = None,
        out: Optional[PaddedLayout] = None,
    ) -> PaddedLayout:
        padded = pad_verify_lens_to_bucket(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=verify_lens.shape[0],
            padded_bs=padded_bs,
        )
        if cap is not None:
            padded = torch.clamp(padded, max=cap)
        indptr = build_qo_indptr(verify_lens=padded)
        result = PaddedLayout(
            verify_lens=padded,
            qo_indptr=indptr.qo_indptr,
            extend_start_loc=indptr.extend_start_loc,
        )
        if out is None:
            return result
        out.verify_lens.copy_(result.verify_lens)
        out.qo_indptr.copy_(result.qo_indptr)
        out.extend_start_loc.copy_(result.extend_start_loc)
        return out

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        padded_bs: int,
        cap: Optional[int] = None,
        out: Optional[PaddedLayout] = None,
    ) -> PaddedLayout:
        bs = verify_lens.shape[0]
        assert padded_bs >= bs, (
            f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
            "batch's requests"
        )
        if out is None:
            device = verify_lens.device
            out = PaddedLayout(
                verify_lens=torch.empty(padded_bs, dtype=torch.int32, device=device),
                qo_indptr=torch.empty(padded_bs + 1, dtype=torch.int32, device=device),
                extend_start_loc=torch.empty(
                    padded_bs, dtype=torch.int32, device=device
                ),
            )
        _pad_verify_layout_kernel[(1,)](
            verify_lens.to(torch.int32).contiguous(),
            out.verify_lens,
            out.qo_indptr,
            out.extend_start_loc,
            bs,
            padded_bs,
            graph_num_tokens,
            0 if cap is None else cap,
            HAS_CAP=cap is not None,
            BLOCK=256,
        )
        return out


@triton.jit(do_not_specialize=["bs", "padded_bs", "graph_num_tokens", "cap"])
def _pad_verify_layout_kernel(
    verify_lens_ptr,
    out_lens_ptr,
    qo_indptr_ptr,
    extend_start_loc_ptr,
    bs,
    padded_bs,
    graph_num_tokens,
    cap,
    HAS_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    real = 0
    for row0 in range(0, bs, BLOCK):
        idx = row0 + tl.arange(0, BLOCK)
        real += tl.sum(tl.load(verify_lens_ptr + idx, mask=idx < bs, other=0), axis=0)
    leftover = graph_num_tokens - real
    num_pad = padded_bs - bs
    num_pad_safe = tl.maximum(num_pad, 1)
    base = leftover // num_pad_safe
    rem = leftover - base * num_pad_safe
    tl.store(qo_indptr_ptr, 0)
    total = 0
    for row0 in range(0, padded_bs, BLOCK):
        idx = row0 + tl.arange(0, BLOCK)
        valid = idx < padded_bs
        is_real = idx < bs
        vl = tl.load(verify_lens_ptr + idx, mask=is_real, other=0)
        pad_len = base + tl.where((idx - bs) < rem, 1, 0)
        lens = tl.where(is_real, vl, pad_len)
        lens = lens + tl.where((num_pad == 0) & (idx == bs - 1), leftover, 0)
        if HAS_CAP:
            lens = tl.minimum(lens, cap)
        lens = tl.where(valid, lens, 0)
        incl = total + tl.cumsum(lens, axis=0)
        tl.store(out_lens_ptr + idx, lens, mask=valid)
        tl.store(qo_indptr_ptr + 1 + idx, incl, mask=valid)
        tl.store(extend_start_loc_ptr + idx, incl - lens, mask=valid)
        total += tl.sum(lens, axis=0)
