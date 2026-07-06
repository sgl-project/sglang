from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_COMPACT_LAYOUT.get()

_SEARCH_NBITS = 11


class CompactRowIndex:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        padded_total: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compact_row_index(
            verify_lens=verify_lens,
            padded_total=padded_total,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        padded_total: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compact_row_index_triton(
            verify_lens=verify_lens,
            padded_total=padded_total,
            device=device,
        )


class CompactVerifyIds:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        layout: RaggedVerifyLayout,
        device: str,
    ) -> torch.Tensor:
        return compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        layout: RaggedVerifyLayout,
        device: str,
    ) -> torch.Tensor:
        return compact_verify_ids_triton(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=device,
        )


def compact_verify_ids(
    *,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    layout: RaggedVerifyLayout,
    device: str,
) -> torch.Tensor:
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    bs = layout.verify_lens.shape[0]
    safe_req = req_id.clamp(max=bs - 1)
    anchors = draft_block_ids[:, 0]
    drafts = draft_tokens[safe_req, (within - 1).clamp_min(0)]
    verify_ids = torch.where(within == 0, anchors[safe_req], drafts)
    verify_ids = torch.where(valid, verify_ids, torch.zeros_like(verify_ids))
    return verify_ids.to(torch.int64)


def compact_row_index(
    *,
    verify_lens: torch.Tensor,
    padded_total: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verify_lens = verify_lens.to(device=device, dtype=torch.int64)
    bs = int(verify_lens.numel())
    incl = torch.cumsum(verify_lens, dim=0)
    start = incl - verify_lens
    real_total = incl[-1]
    row = torch.arange(padded_total, device=device, dtype=torch.int64)
    valid = row < real_total
    req_id = torch.searchsorted(incl, row, right=True)
    req_id = torch.where(valid, req_id, torch.full_like(req_id, bs))
    within = torch.where(
        valid, row - start[req_id.clamp(max=bs - 1)], torch.zeros_like(row)
    )
    return req_id, within, valid


@triton.jit
def _compact_row_index_kernel(
    incl_ptr,
    req_out_ptr,
    within_out_ptr,
    valid_out_ptr,
    bs,
    n,
    BLOCK: tl.constexpr,
    NBITS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    row = offs.to(tl.int64)
    real_total = tl.load(incl_ptr + (bs - 1))
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.full([BLOCK], bs, dtype=tl.int32)
    for _ in range(NBITS):
        mid = (lo + hi) // 2
        active = lo < hi
        val = tl.load(incl_ptr + tl.minimum(mid, bs - 1), mask=mask, other=0)
        go_right = val <= row
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & (~go_right), mid, hi)
    req = lo
    gidx = tl.maximum(req - 1, 0)
    start = tl.load(incl_ptr + gidx, mask=mask, other=0)
    start = tl.where(req > 0, start, 0)
    valid = row < real_total
    within = tl.where(valid, row - start, 0)
    req_final = tl.where(valid, req.to(tl.int64), bs)
    tl.store(req_out_ptr + offs, req_final, mask=mask)
    tl.store(within_out_ptr + offs, within, mask=mask)
    tl.store(valid_out_ptr + offs, valid, mask=mask)


def compact_row_index_triton(
    *,
    verify_lens: torch.Tensor,
    padded_total: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verify_lens = verify_lens.to(device=device, dtype=torch.int64).contiguous()
    bs = verify_lens.shape[0]
    incl = torch.cumsum(verify_lens, dim=0).contiguous()
    req = torch.empty(padded_total, dtype=torch.int64, device=device)
    within = torch.empty(padded_total, dtype=torch.int64, device=device)
    valid = torch.empty(padded_total, dtype=torch.bool, device=device)
    BLOCK = 256
    grid = (triton.cdiv(padded_total, BLOCK),)
    _compact_row_index_kernel[grid](
        incl, req, within, valid, bs, padded_total, BLOCK=BLOCK, NBITS=_SEARCH_NBITS
    )
    return req, within, valid


@triton.jit
def _compact_verify_ids_gather_kernel(
    req_ptr,
    within_ptr,
    draft_block_ids_ptr,
    draft_tokens_ptr,
    out_ptr,
    bs,
    gamma,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    req = tl.load(req_ptr + offs, mask=mask, other=0)
    within = tl.load(within_ptr + offs, mask=mask, other=0)
    valid = req < bs
    safe_req = tl.minimum(req, bs - 1)
    anchor = tl.load(draft_block_ids_ptr + safe_req * gamma, mask=mask, other=0)
    wcol = tl.maximum(within - 1, 0)
    draft = tl.load(draft_tokens_ptr + safe_req * gamma + wcol, mask=mask, other=0)
    v = tl.where(within == 0, anchor, draft)
    v = tl.where(valid, v, 0)
    tl.store(out_ptr + offs, v.to(tl.int64), mask=mask)


def compact_verify_ids_triton(
    *,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    layout: RaggedVerifyLayout,
    device: str,
) -> torch.Tensor:
    req, within, _valid = compact_row_index_triton(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    bs = layout.verify_lens.shape[0]
    gamma = draft_tokens.shape[1]
    draft_block_ids = draft_block_ids.to(device=device, dtype=torch.int64).contiguous()
    draft_tokens = draft_tokens.to(device=device, dtype=torch.int64).contiguous()
    n = layout.graph_num_tokens
    out = torch.empty(n, dtype=torch.int64, device=device)
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _compact_verify_ids_gather_kernel[grid](
        req, within, draft_block_ids, draft_tokens, out, bs, gamma, n, BLOCK=BLOCK
    )
    return out
