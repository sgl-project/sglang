from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_QO_INDPTR.get()


class QoIndptrResult(msgspec.Struct):
    qo_indptr: torch.Tensor
    extend_start_loc: torch.Tensor


class BuildQoIndptr:
    @classmethod
    def execute(cls, *args, **kwargs) -> QoIndptrResult:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

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
