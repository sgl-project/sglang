"""Decode LoRA shrink+expand in one launch.

At decode (one LoRA segment of <= 16 tokens under the CUDA graph) each dense
LoRA'd module runs a chain of tiny latency-bound kernels: split-K shrink
partials, their reduction, and the expand (gate|up, q|kv, plain). Each launch
costs a few microseconds of dependent metadata loads and ramp for almost no
work. The kernels here do the chain in one launch with the stock kernels'
arithmetic (same tiles, same fp32 accumulation order, same bf16 rounding
points), so the results are bit-identical to the chain they replace:

* ``splitk_ab`` (A with K >= 2048, then an n-slice expand): every program
  computes one fp32 split-K partial like ``_sgemm_lora_a_splitk_partial_kernel``
  and counts itself in; once all partials are in, every program sums the shrink
  columns it needs in split order and rounds them like
  ``_sgemm_lora_a_splitk_reduce_kernel``, and applies its share of the expand
  tiles like ``_gate_up_lora_b_kernel`` / ``_qkv_lora_b_kernel``. The last
  program to leave resets the counters. All programs of a segment wait on each
  other, so the grid (n_rblocks x KSPLIT per segment, a few dozen programs)
  must be co-resident, which it is by far.
* ``small_k_ab`` (A with small K, then one expand, e.g. a row-parallel down
  projection whose partial output the caller reduces): every expand program
  recomputes the [tokens, r] shrink like ``_sgemm_lora_a_kernel`` (a few KB of
  A) and applies its output tile like ``_sgemm_lora_b_kernel``. No
  cross-program dependency.

Enabled by SGLANG_OPT_LORA_DECODE_FUSED_LAUNCHES, together with the decode split-K
LoRA-A (sgemm_lora_a) and the LoRA-A side-stream overlap of the triton backend.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.kernel_utils import _resolve_token_positions
from sglang.srt.environ import envs
from sglang.srt.lora.utils import LoRABatchInfo

DECODE_FUSED_LAUNCHES = envs.SGLANG_OPT_LORA_DECODE_FUSED_LAUNCHES.get()
_MAX_DECODE_TOKENS = 16  # one BLOCK_S tile
_SMALLK_MAX_K = 1024

_workspaces = {}


def _workspace(key, ksplit, rows, r, nseg, device):
    """Per call-site fp32 partials and zeroed per-segment counters (the last program out resets them)."""
    ws = _workspaces.get(key)
    if (
        ws is None
        or ws[0].shape[0] < ksplit
        or ws[0].shape[1] < rows
        or ws[0].shape[2] < r
        or ws[1].shape[0] < nseg
    ):
        ws = (
            torch.empty((ksplit, rows, r), dtype=torch.float32, device=device),
            torch.zeros((max(nseg, 1), 4), dtype=torch.int32, device=device),
        )
        _workspaces[key] = ws
    return ws


@triton.jit
def _splitk_ab_kernel(
    x,
    wa,
    partial,
    ticket,
    wb,
    out,
    n_offs,
    R,
    K,
    x_s0,
    x_s1,
    wa_s0,
    wa_s1,
    wa_s2,
    p_sk,
    p_ss,
    wb_s0,
    wb_s1,
    wb_s2,
    o_s0,
    o_s1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    scalings,
    NSLICE: tl.constexpr,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    N_RBLOCKS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    OUT_TILES: tl.constexpr,
    KSPLIT: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    stack_num: tl.constexpr = NSLICE
    N = tl.minimum(R, rank * stack_num)
    s_offset = tl.arange(0, BLOCK_S)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    s_mask = s_offset < seg_len
    # ---- phase 1: this program's split-K partial (as _sgemm_lora_a_splitk_partial_kernel) ----
    if pid_n * BLOCK_N < N:
        n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
        k_offset = tl.arange(0, BLOCK_K) + pid_k * BLOCK_K
        x_tile = tl.load(
            x + s_physical[:, None] * x_s0 + k_offset[None, :] * x_s1,
            mask=s_mask[:, None] & (k_offset[None, :] < K),
            other=0.0,
        )
        w_tile = tl.load(
            wa
            + w_index * wa_s0
            + k_offset[:, None] * wa_s2
            + n_offset[None, :] * wa_s1,
            mask=(k_offset[:, None] < K) & (n_offset[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(x_tile, w_tile)
        tl.store(
            partial + pid_k * p_sk + s_physical[:, None] * p_ss + n_offset[None, :],
            acc,
            mask=s_mask[:, None] & (n_offset[None, :] < N),
        )
    # ---- arrival: every program of the segment waits until all partials are in ----
    tl.debug_barrier()
    total = N_RBLOCKS * KSPLIT
    cnt = ticket + batch_id * 4  # [arrive, depart]
    tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
    while tl.load(cnt, volatile=True) < total:
        pass
    tl.atomic_add(cnt, 0, sem="acquire", scope="gpu")
    # ---- each program expands its share of the BLOCK_OUT-wide output tiles (as _gate_up_lora_b_kernel /
    # _qkv_lora_b_kernel, K of the expand = rank). The slice's shrink columns are summed from the partials in
    # split order and rounded (as _sgemm_lora_a_splitk_reduce_kernel), per program: a few KB, no second round trip.
    scaling = tl.load(scalings + w_index)
    KR = tl.minimum(BLOCK_R, rank)
    k_offset = tl.arange(0, BLOCK_R)
    c = pid_n + N_RBLOCKS * pid_k
    for tile in range(c, NSLICE * OUT_TILES, total):
        sl = tile // OUT_TILES
        ob = tile % OUT_TILES
        cols = sl * KR + k_offset
        cmask = s_mask[:, None] & (k_offset[None, :] < KR)
        sh = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
        for kk in tl.static_range(KSPLIT):
            sh += tl.load(
                partial + kk * p_sk + s_physical[:, None] * p_ss + cols[None, :],
                mask=cmask,
                other=0.0,
                cache_modifier=".cg",
            )
        x_tile = sh.to(x.dtype.element_ty)
        n_start = tl.load(n_offs + sl)
        n_size = tl.load(n_offs + sl + 1) - n_start
        n_offset = tl.arange(0, BLOCK_OUT) + ob * BLOCK_OUT
        w_tile = tl.load(
            wb
            + w_index * wb_s0
            + n_start * wb_s1
            + k_offset[:, None] * wb_s2
            + n_offset[None, :] * wb_s1,
            mask=(k_offset[:, None] < KR) & (n_offset[None, :] < n_size),
            other=0.0,
        )
        acc = tl.zeros((BLOCK_S, BLOCK_OUT), dtype=tl.float32)
        acc += tl.dot(x_tile, w_tile)
        acc *= scaling
        res = acc.to(x.dtype.element_ty)
        o_ptr = (
            out + n_start * o_s1 + s_physical[:, None] * o_s0 + n_offset[None, :] * o_s1
        )
        o_mask = s_mask[:, None] & (n_offset[None, :] < n_size)
        res += tl.load(o_ptr, mask=o_mask)
        tl.store(o_ptr, res, mask=o_mask)
    # ---- the last program to leave resets the segment's counters for the next call ----
    d = tl.atomic_add(cnt + 1, 1, sem="acq_rel", scope="gpu")
    if d == total - 1:
        tl.atomic_xchg(cnt, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(cnt + 1, 0, sem="relaxed", scope="gpu")


def splitk_ab_eligible(
    x: torch.Tensor, weights_a: torch.Tensor, batch_info: LoRABatchInfo
) -> bool:
    return (
        DECODE_FUSED_LAUNCHES
        and batch_info.max_len is not None
        and batch_info.max_len <= _MAX_DECODE_TOKENS
        and x.shape[0] <= _MAX_DECODE_TOKENS
        and weights_a.shape[-1] >= 2048
        and x.dtype == torch.bfloat16
    )


def splitk_ab_fwd(
    x: torch.Tensor,  # (s, K)
    weights_a: torch.Tensor,  # (num_lora, nslice * r, K)
    weights_b: torch.Tensor,  # (num_lora, sum of slice dims, r)
    batch_info: LoRABatchInfo,
    n_offs: torch.Tensor,  # int32 (nslice + 1,) slice offsets into the output / B rows
    max_slice_dim: int,
    nslice: int,
    base_output: torch.Tensor,
) -> torch.Tensor:
    assert x.is_contiguous() and weights_a.is_contiguous() and weights_b.is_contiguous()
    S, K = x.shape
    R = weights_a.shape[-2]
    r = weights_b.shape[-1]
    assert R == nslice * r and r <= 16
    BLOCK_S, BLOCK_N, BLOCK_K, BLOCK_R = 16, 16, 256, 16
    ksplit = triton.cdiv(K, BLOCK_K)
    n_rblocks = triton.cdiv(R, BLOCK_N)
    block_out = 64  # the stock gate|up / qkv expand tile
    out_tiles = triton.cdiv(max_slice_dim, block_out)
    partial, ticket = _workspace(
        ("splitk_ab", weights_a.data_ptr()),
        ksplit,
        _MAX_DECODE_TOKENS,
        n_rblocks * BLOCK_N,
        batch_info.bs,
        x.device,
    )
    sorted_by_adapter = batch_info.permutation is not None
    _splitk_ab_kernel[(n_rblocks, batch_info.bs, ksplit)](
        x,
        weights_a,
        partial,
        ticket,
        weights_b,
        base_output,
        n_offs,
        R,
        K,
        x.stride(0),
        x.stride(1),
        weights_a.stride(0),
        weights_a.stride(1),
        weights_a.stride(2),
        partial.stride(0),
        partial.stride(1),
        weights_b.stride(0),
        weights_b.stride(1),
        weights_b.stride(2),
        base_output.stride(0),
        base_output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        NSLICE=nslice,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_R=BLOCK_R,
        N_RBLOCKS=n_rblocks,
        BLOCK_OUT=block_out,
        OUT_TILES=out_tiles,
        KSPLIT=ksplit,
    )
    return base_output


@triton.jit
def _small_k_ab_kernel(
    x,
    wa,
    wb,
    out,
    N,
    K,
    R,
    x_s0,
    x_s1,
    wa_s0,
    wa_s1,
    wa_s2,
    wb_s0,
    wb_s1,
    wb_s2,
    o_s0,
    o_s1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    scalings,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    s_offset = tl.arange(0, BLOCK_S)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    s_mask = s_offset < seg_len
    # shrink (as _sgemm_lora_a_kernel with BLOCK_N = BLOCK_R, stack_num 1)
    NA = tl.minimum(R, rank)
    r_offset = tl.arange(0, BLOCK_R)
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = x + s_physical[:, None] * x_s0 + k_offset[None, :] * x_s1
    a_ptrs = (
        wa + w_index * wa_s0 + k_offset[:, None] * wa_s2 + r_offset[None, :] * wa_s1
    )
    shrink = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=s_mask[:, None] & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        a_tile = tl.load(
            a_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (r_offset[None, :] < NA),
            other=0.0,
        )
        shrink += tl.dot(x_tile, a_tile)
        x_ptrs += BLOCK_K * x_s1
        a_ptrs += BLOCK_K * wa_s2
    lora_a = shrink.to(x.dtype.element_ty)
    # expand tile (as _sgemm_lora_b_kernel, one k step of BLOCK_R = r)
    KB = tl.minimum(BLOCK_R, rank)
    lora_a = tl.where(r_offset[None, :] < KB, lora_a, 0.0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    n_mask = n_offset[None, :] < N
    w_tile = tl.load(
        wb + w_index * wb_s0 + r_offset[:, None] * wb_s2 + n_offset[None, :] * wb_s1,
        mask=(r_offset[:, None] < KB) & n_mask,
        other=0.0,
    )
    acc = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    acc += tl.dot(lora_a, w_tile)
    acc *= scaling
    res = acc.to(x.dtype.element_ty)
    o_ptr = out + s_physical[:, None] * o_s0 + n_offset[None, :] * o_s1
    o_mask = s_mask[:, None] & n_mask
    res += tl.load(o_ptr, mask=o_mask, other=0.0)
    tl.store(o_ptr, res, mask=o_mask)


def small_k_ab_eligible(
    x: torch.Tensor,
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output,
) -> bool:
    return (
        DECODE_FUSED_LAUNCHES
        and base_output is not None
        and batch_info.max_len is not None
        and batch_info.max_len <= _MAX_DECODE_TOKENS
        and x.shape[0] <= _MAX_DECODE_TOKENS
        and weights_a.shape[-1] <= _SMALLK_MAX_K
        and weights_a.shape[-2] <= 16
        and weights_b.shape[-1] == weights_a.shape[-2]
        and x.dtype == torch.bfloat16
    )


def small_k_ab_fwd(x, weights_a, weights_b, batch_info: LoRABatchInfo, base_output):
    assert x.is_contiguous() and weights_a.is_contiguous() and weights_b.is_contiguous()
    S, K = x.shape
    R = weights_a.shape[-2]
    N = weights_b.shape[-2]
    BLOCK_S, BLOCK_N, BLOCK_K, BLOCK_R = 16, 256, 256, 16
    sorted_by_adapter = batch_info.permutation is not None
    _small_k_ab_kernel[(triton.cdiv(N, BLOCK_N), batch_info.bs)](
        x,
        weights_a,
        weights_b,
        base_output,
        N,
        K,
        R,
        x.stride(0),
        x.stride(1),
        weights_a.stride(0),
        weights_a.stride(1),
        weights_a.stride(2),
        weights_b.stride(0),
        weights_b.stride(1),
        weights_b.stride(2),
        base_output.stride(0),
        base_output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_R=BLOCK_R,
    )
    return base_output


def lora_a_overlap_eligible(
    x: torch.Tensor, weights_a: torch.Tensor, batch_info: LoRABatchInfo
) -> bool:
    """Decode LoRA-A that may run on a side stream next to its base GEMM
    (e.g. a row-parallel o_proj or a replicated q_a|kv_a projection)."""
    return (
        DECODE_FUSED_LAUNCHES
        and batch_info.max_len is not None
        and batch_info.max_len <= _MAX_DECODE_TOKENS
        and x.dim() == 2
        and x.shape[0] <= _MAX_DECODE_TOKENS
        and x.is_contiguous()
        and weights_a.is_contiguous()
    )
