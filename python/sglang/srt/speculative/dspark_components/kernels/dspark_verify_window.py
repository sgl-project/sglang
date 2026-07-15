from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.memory.req_to_token_pool import assign_extend_cache_locs_func
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.kernels.dispatch import inputs_on_cuda
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor


class BuildRaggedVerifyWindow:
    @classmethod
    def execute(cls, *args, **kwargs) -> RaggedVerifyWindow:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )

    @classmethod
    def triton(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window_triton(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )


def build_ragged_verify_window(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, valid = compact_row_index(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    safe_req = req_id.clamp(max=bs - 1)
    positions = torch.where(
        valid,
        prefix_lens.to(torch.int64)[safe_req] + within,
        torch.zeros_like(within),
    )
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    verify_cache_loc = torch.nn.functional.pad(
        real_cache_loc, (0, padded_total - real_cache_loc.shape[0])
    )
    verify_cache_loc = torch.where(
        valid, verify_cache_loc, torch.zeros_like(verify_cache_loc)
    )

    verify_ids = compact_verify_ids(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )

    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )


@triton.jit
def _ragged_finalize_kernel(
    req_ptr,
    within_ptr,
    prefix_ptr,
    cache_ptr,
    pos_out_ptr,
    cache_out_ptr,
    bs,
    n,
    real_len,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    req = tl.load(req_ptr + offs, mask=mask, other=0)
    within = tl.load(within_ptr + offs, mask=mask, other=0)
    valid = req < bs
    safe_req = tl.minimum(req, bs - 1)
    prefix = tl.load(prefix_ptr + safe_req, mask=mask, other=0)
    pos = tl.where(valid, prefix + within, 0)
    lmask = mask & (offs < real_len)
    cl = tl.load(cache_ptr + offs, mask=lmask, other=0)
    cl = tl.where(valid, cl, 0)
    tl.store(pos_out_ptr + offs, pos, mask=mask)
    tl.store(cache_out_ptr + offs, cl, mask=mask)


def build_ragged_verify_window_triton(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, _valid = compact_row_index_triton(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    prefix_i64 = prefix_lens.to(device=device, dtype=torch.int64).contiguous()
    positions = torch.empty(padded_total, dtype=torch.int64, device=device)
    verify_cache_loc = torch.empty(
        padded_total, dtype=real_cache_loc.dtype, device=device
    )
    BLOCK = 256
    grid = (triton.cdiv(padded_total, BLOCK),)
    _ragged_finalize_kernel[grid](
        req_id,
        within,
        prefix_i64,
        real_cache_loc,
        positions,
        verify_cache_loc,
        bs,
        padded_total,
        real_cache_loc.shape[0],
        BLOCK=BLOCK,
    )

    verify_ids = compact_verify_ids_triton(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )


_SEARCH_NBITS = 11


class CompactRowIndex:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

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
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

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


class ScatterCompactToStrided:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        compact: torch.Tensor,
        layout: RaggedVerifyLayout,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        compact: torch.Tensor,
        layout: RaggedVerifyLayout,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided_triton(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )


def scatter_compact_to_strided(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    compact = compact[: layout.graph_num_tokens]
    strided = torch.full(
        (bs * stride + 1, dim), fill_value, dtype=compact.dtype, device=device
    )
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    sink = bs * stride
    strided_pos = torch.where(
        valid,
        req_id.clamp(max=bs - 1) * stride + within,
        torch.full_like(within, sink),
    )
    strided.index_copy_(0, strided_pos, compact)
    return strided[: bs * stride]


@triton.jit
def _scatter_compact_to_strided_kernel(
    compact_ptr,
    verify_lens_ptr,
    start_ptr,
    out_ptr,
    stride,
    dim,
    fill_value,
    BLOCK_D: tl.constexpr,
):
    o = tl.program_id(0).to(tl.int64)
    dblk = tl.program_id(1)
    i = o // stride
    w = o % stride
    vl_i = tl.load(verify_lens_ptr + i)
    start_i = tl.load(start_ptr + i)
    d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = d < dim
    in_range = w < vl_i
    src = tl.where(in_range, start_i + w, 0)
    val = tl.load(compact_ptr + src * dim + d, mask=dmask & in_range, other=0)
    val = tl.where(in_range, val, fill_value)
    tl.store(out_ptr + o * dim + d, val, mask=dmask)


def scatter_compact_to_strided_into(
    *,
    compact: torch.Tensor,
    verify_lens: torch.Tensor,
    out: torch.Tensor,
    stride: int,
    fill_value: float,
) -> torch.Tensor:
    dim = compact.shape[1]
    fill_value = float(fill_value) if out.dtype.is_floating_point else int(fill_value)
    compact = compact.contiguous()
    verify_lens = verify_lens.to(dtype=torch.int64).contiguous()
    start = (torch.cumsum(verify_lens, dim=0) - verify_lens).contiguous()
    n_out = out.shape[0]
    BLOCK_D = 1024
    grid = (n_out, triton.cdiv(dim, BLOCK_D))
    _scatter_compact_to_strided_kernel[grid](
        compact,
        verify_lens,
        start,
        out,
        stride,
        dim,
        fill_value,
        BLOCK_D=BLOCK_D,
    )
    return out


def scatter_compact_to_strided_triton(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    out = torch.empty((bs * stride, dim), dtype=compact.dtype, device=device)
    return scatter_compact_to_strided_into(
        compact=compact,
        verify_lens=layout.verify_lens.to(device=device),
        out=out,
        stride=stride,
        fill_value=fill_value,
    )


class CommitInjectLayoutResult(msgspec.Struct):
    swa_loc: torch.Tensor
    positions: torch.Tensor


class BuildCommitInjectLayout:
    @classmethod
    def execute(cls, *args, **kwargs) -> CommitInjectLayoutResult:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_pos_offsets: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_pos_offsets: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout_triton(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )


def build_commit_inject_layout(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    from sglang.kernels.ops.memory.req_to_token_pool import (
        assign_extend_cache_locs_func,
    )

    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device

    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(dtype=torch.int64)

    cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + stride,
        batch_size=bs,
        draft_token_num=stride,
        device=device,
    ).to(dtype=torch.int64)
    swa_loc = full_to_swa_mapping[cache_loc].to(torch.int32)

    col = torch.arange(stride, device=device).view(1, -1)
    committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
    swa_loc = torch.where(committed, swa_loc, torch.full_like(swa_loc, -1))

    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


@triton.jit
def _commit_inject_layout_kernel(
    req_pool_ptr,
    req_to_token_ptr,
    prefix_lens_ptr,
    block_pos_offsets_ptr,
    full_to_swa_ptr,
    commit_lens_ptr,
    swa_loc_ptr,
    positions_ptr,
    rt_stride,
    stride,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = offs // stride
    c = offs % stride

    prefix = tl.load(prefix_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    pos_off = tl.load(block_pos_offsets_ptr + c, mask=mask, other=0).to(tl.int64)
    rp = tl.load(req_pool_ptr + r, mask=mask, other=0).to(tl.int64)
    full_loc = tl.load(
        req_to_token_ptr + rp * rt_stride + prefix + pos_off, mask=mask, other=0
    ).to(tl.int64)
    swa = tl.load(full_to_swa_ptr + full_loc, mask=mask, other=-1).to(tl.int32)

    commit_len = tl.load(commit_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    swa = tl.where(c.to(tl.int64) < commit_len, swa, -1)

    tl.store(swa_loc_ptr + offs, swa, mask=mask)
    tl.store(positions_ptr + offs, prefix + pos_off, mask=mask)


def build_commit_inject_layout_triton(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    bs = req_pool_indices.shape[0]
    n = bs * stride
    device = req_pool_indices.device

    swa_loc = torch.empty(n, dtype=torch.int32, device=device)
    positions = torch.empty(n, dtype=torch.int64, device=device)
    BLOCK = 256
    _commit_inject_layout_kernel[(triton.cdiv(n, BLOCK),)](
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa_mapping,
        commit_lens,
        swa_loc,
        positions,
        req_to_token.stride(0),
        stride,
        n,
        BLOCK=BLOCK,
    )
    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


class BuildOutTokens:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

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
