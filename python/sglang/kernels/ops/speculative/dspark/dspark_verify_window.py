from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import swa_ring_rows
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.utils import (
    is_npu,
)

_is_npu = is_npu()


class RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor
    # Each packed lane's flat position in the `[bs, verify_num_draft_tokens]`
    # verify window, -1 on a padded lane.
    window_index: torch.Tensor


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
        window_index=torch.where(
            valid, safe_req.to(torch.int64) * verify_num_draft_tokens + within, -1
        ),
    )


@triton.jit(
    do_not_specialize=[
        "req_to_token_stride",
        "draft_block_ids_stride",
        "draft_tokens_stride",
        "num_rows",
        "bs",
        "n",
        "width",
    ]
)
def _ragged_verify_window_kernel(
    verify_lens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    req_to_token_stride,
    draft_block_ids_ptr,
    draft_block_ids_stride,
    draft_tokens_ptr,
    draft_tokens_stride,
    positions_ptr,
    cache_loc_ptr,
    verify_ids_ptr,
    window_index_ptr,
    num_rows,
    bs,
    n,
    width,
    BLOCK_T: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offs < n
    req = tl.zeros([BLOCK_T], dtype=tl.int32)
    start = tl.zeros([BLOCK_T], dtype=tl.int32)
    total = 0
    for row0 in range(0, num_rows, BLOCK_BS):
        rows = row0 + tl.arange(0, BLOCK_BS)
        lens = tl.load(verify_lens_ptr + rows, mask=rows < num_rows, other=0)
        lens = lens.to(tl.int32)
        incl = total + tl.cumsum(lens, axis=0)
        before = incl[None, :] <= offs[:, None]
        req += tl.sum(before.to(tl.int32), axis=1)
        start += tl.sum(tl.where(before, lens[None, :], 0), axis=1)
        total += tl.sum(lens, axis=0)
    valid = mask & (offs < total)
    safe_req = tl.minimum(req, bs - 1)
    within = tl.where(valid, offs - start, 0)
    prefix = tl.load(prefix_lens_ptr + safe_req, mask=mask, other=0).to(tl.int64)
    pool_row = tl.load(req_pool_indices_ptr + safe_req, mask=mask, other=0).to(tl.int64)
    cache_loc = tl.load(
        req_to_token_ptr + pool_row * req_to_token_stride + prefix + within,
        mask=valid,
        other=0,
    )
    anchor = tl.load(
        draft_block_ids_ptr + safe_req.to(tl.int64) * draft_block_ids_stride,
        mask=mask,
        other=0,
    )
    draft = tl.load(
        draft_tokens_ptr
        + safe_req.to(tl.int64) * draft_tokens_stride
        + tl.maximum(within - 1, 0),
        mask=mask,
        other=0,
    )
    verify_id = tl.where(within == 0, anchor.to(tl.int64), draft.to(tl.int64))
    tl.store(positions_ptr + offs, tl.where(valid, prefix + within, 0), mask=mask)
    tl.store(
        cache_loc_ptr + offs, tl.where(valid, cache_loc.to(tl.int64), 0), mask=mask
    )
    tl.store(verify_ids_ptr + offs, tl.where(valid, verify_id, 0), mask=mask)
    tl.store(
        window_index_ptr + offs,
        tl.where(valid, safe_req.to(tl.int64) * width + within, -1),
        mask=mask,
    )


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
    req_to_token = model_runner.req_to_token_pool.req_to_token
    verify_lens = layout.verify_lens.contiguous()
    num_rows = verify_lens.shape[0]
    padded_total = layout.graph_num_tokens
    assert num_rows == bs and draft_tokens.stride(1) == 1
    positions = torch.empty(padded_total, dtype=torch.int64, device=device)
    verify_cache_loc = torch.empty_like(positions)
    verify_ids = torch.empty_like(positions)
    window_index = torch.empty_like(positions)
    _ragged_verify_window_kernel[(triton.cdiv(padded_total, 32),)](
        verify_lens,
        batch.seq_lens,
        batch.req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        draft_block_ids,
        draft_block_ids.stride(0),
        draft_tokens,
        draft_tokens.stride(0),
        positions,
        verify_cache_loc,
        verify_ids,
        window_index,
        num_rows,
        bs,
        padded_total,
        verify_num_draft_tokens,
        BLOCK_T=32,
        BLOCK_BS=256,
    )
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
        window_index=window_index,
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


def build_ring_commit_inject_layout(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
    ring_stride: int,
) -> CommitInjectLayoutResult:
    """Per-request SWA ring counterpart of build_commit_inject_layout.

    Paged injection translates the verify tokens' full cache locs through
    ``full_to_swa_mapping``; a per-request ring is addressed directly by
    `swa_ring_rows`, so compute the ring row here instead. Uncommitted tokens
    (col >= commit_len) get loc = -1 and are skipped by the scatter. All ops are
    static-shape (CUDA-graph safe).
    """
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(torch.int64)
    state_slot = (
        req_pool_indices.to(torch.int64).view(-1, 1).expand(bs, stride).reshape(-1)
    )
    loc = swa_ring_rows(state_slot, positions, ring_stride)
    col = torch.arange(stride, device=device).view(1, -1)
    committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
    swa_loc = torch.where(committed, loc, torch.full_like(loc, -1)).to(torch.int32)
    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


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
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(dtype=torch.int64)
    col = torch.arange(stride, device=device).view(1, -1)
    committed = col < commit_lens.to(torch.long).view(-1, 1)
    req_rows = req_pool_indices.to(torch.long).view(-1, 1).expand(bs, stride)
    rows = torch.where(committed, req_rows, 0)
    cols = torch.where(committed, positions_2d.long(), 0)
    cache_loc = torch.where(committed, req_to_token[rows, cols].long(), 0)
    swa_loc = torch.where(committed, full_to_swa_mapping[cache_loc], -1)
    return CommitInjectLayoutResult(
        swa_loc=swa_loc.to(torch.int32).reshape(-1), positions=positions
    )


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
    # Padded/uncommitted slots need no cache lookup. Their request-table
    # entries may be stale or uninitialized, so mask before dereferencing them.
    commit_len = tl.load(commit_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    committed = mask & (c.to(tl.int64) < commit_len)
    full_loc = tl.load(
        req_to_token_ptr + rp * rt_stride + prefix + pos_off,
        mask=committed,
        other=0,
    ).to(tl.int64)
    swa = tl.load(full_to_swa_ptr + full_loc, mask=committed, other=-1).to(tl.int32)

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
        if _is_npu:
            from sgl_kernel_npu.operators.sample import (
                build_out_tokens_triton as npu_build_out_tokens_triton,
            )

            return npu_build_out_tokens_triton(
                draft_tokens=draft_tokens,
                correct_len=correct_len,
                bonus=bonus,
                verify_num_draft_tokens=verify_num_draft_tokens,
                gamma=gamma,
            )
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
    """Return accepted drafts, the bonus token, and zero-filled tail slots."""
    bs = draft_tokens.shape[0]
    out_tokens = torch.zeros(
        (bs, verify_num_draft_tokens),
        dtype=torch.int64,
        device=draft_tokens.device,
    )
    correct_len = correct_len.to(torch.int64)
    keep = (
        torch.arange(gamma, device=draft_tokens.device)[None, :] < correct_len[:, None]
    )
    out_tokens[:, :gamma] = torch.where(keep, draft_tokens.to(torch.int64), 0)
    out_tokens.scatter_(1, correct_len[:, None], bonus[:, None])
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
    # Rejected drafts must not reach downstream token gathers.
    val = tl.where(k == cl, bonus, tl.where(k < cl, draft, 0))
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
