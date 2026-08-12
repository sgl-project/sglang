# SPDX-License-Identifier: Apache-2.0
# Kimi-K3 Attention Residual: snapshot bank + aggregation.
#
# The public API is the AttnResidual class (constructed once per forward pass).
# It owns the frozen snapshot bank [T, NB, H] and the valid-row counter, and
# dispatches each aggregation point (score rows → softmax → weighted sum →
# RMSNorm) by hardware capability:
#   fast  — warp-specialized TMA kernel: cp.async.bulk producer +
#           online-softmax consumers over a double-buffered chunk ring, out
#           norm fused, per-nvb tuned launch config, one persistent CTA per
#           SM. Taken on SM100+ with H=7168.
#   hip   — single Triton kernel, everything in one launch; taken on ROCm
#           within its register budget.
#   fused — Triton 2-kernel pipeline with full H-parallelism; the fallback
#           everywhere the fast kernel does not apply.
# aggregate_stream_torch is the eager reference (tests and the
# H % _BLOCK_H != 0 shape fallback of aggregate_stream).

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import is_hip

_BLOCK_H: int = 1024  # H = 7168 = 7 x 1024
_MAX_ROWS: int = 16  # next_pow2(8 + 1), K3 has <= 8 snapshots

_FAST_SUPPORTED = None
_HIP_SHAPE_GATE = None


def _use_fast(hidden_size: int) -> bool:
    """The TMA kernel needs SM100+ (tcgen05, cp.async.bulk) and its H=7168
    template instantiation; everything else takes the triton pipeline."""
    global _FAST_SUPPORTED
    if _FAST_SUPPORTED is None:
        major, _ = torch.cuda.get_device_capability()
        _FAST_SUPPORTED = major >= 10
    return _FAST_SUPPORTED and hidden_size == 7168


def _use_hip_fused(hidden_size: int, nvb: int) -> bool:
    """This gate picks the single-kernel ROCm Triton kernel instead of the
    2-kernel pipeline."""
    if not is_hip():
        return False
    global _HIP_SHAPE_GATE
    if _HIP_SHAPE_GATE is None:
        from sglang.kernels.ops.kimi_k3.attn_res_hip import supports_attn_res_hip

        _HIP_SHAPE_GATE = supports_attn_res_hip
    return _HIP_SHAPE_GATE(hidden_size, nvb)


def get_cw(
    proj: ReplicatedLinear,
    norm: RMSNorm,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cached product norm_weight ⊙ proj_weight (both [H]) in `dtype`.

    Cached per dtype: the fast kernel consumes bf16 while the triton path
    consumes fp32, and a shared slot would hand one path the other's dtype."""
    cache = getattr(proj, "_attn_res_cw_cache", None)
    if cache is None:
        cache = {}
        proj._attn_res_cw_cache = cache
    cw = cache.get(dtype)
    if cw is None:
        cw = (norm.weight.float() * proj.weight.squeeze().float()).contiguous()
        cw = cache[dtype] = cw.to(dtype)
    return cw


def _aggregate_fast(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    write_bank_row: bool = False,
) -> torch.Tensor:
    """Warp-specialized TMA kernel: online softmax over row chunks with the
    output RMSNorm fused, one persistent CTA per SM, per-nvb tuned launch
    config (GB300 benchmark winner across nvb). With write_bank_row the kernel
    also snapshots the prefix row into bank[:, nvb, :] (bit-exact, zero extra
    reads — the row streams through the score pass anyway)."""
    from sglang.kernels.ops.kimi_k3.attn_res import attn_res_fused_tma

    # The kernel applies one eps to both the score norm and the output norm.
    assert score_norm.variance_epsilon == out_norm.variance_epsilon

    cw = get_cw(score_proj, score_norm, dtype=torch.bfloat16)
    out = torch.empty_like(prefix_sum)
    attn_res_fused_tma(
        prefix_sum,
        bank,
        cw,
        out_norm.weight,
        out,
        nvb,
        score_norm.variance_epsilon,
        write_prefix=write_bank_row,
    )
    return out


@triton.jit
def _score_kernel(
    prefix_ptr,  # [T, H]
    bank_ptr,  # [T, NB_total, H]
    cw_ptr,  # [H] fp32
    scores_ptr,  # [T, MAX_ROWS] fp32
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One CTA per (token, row): scan H, output one scalar score."""
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    if j > NVB:
        return
    sumsq = 0.0
    dotv = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(
                tl.float32
            )
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        cw = tl.load(cw_ptr + offs_h)
        sumsq += tl.sum(v * v)
        dotv += tl.sum(v * cw)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + j, dotv * rrms)


@triton.jit
def _combine_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,  # [T, MAX_ROWS] fp32
    out_ptr,  # [T, H]
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """One CTA per (token, H-chunk): softmax(scores) → weighted sum → write chunk.

    Softmax is redundantly computed by each H-chunk CTA (≤16 elements, trivial).
    This gives full H-parallelism: 7 CTAs for H=7168/1024.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h0 = pid_h * BLOCK_H

    # Softmax (redundant per chunk, 16 fp32 ops)
    offs_b = tl.arange(0, MAX_ROWS)
    mask_b = offs_b <= NVB
    raw = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b, mask=mask_b, other=float("-inf")
    )
    m = tl.max(raw, axis=0)
    e = tl.where(mask_b, tl.exp(raw - m), 0.0)
    p = e / tl.sum(e, axis=0)

    # Weighted sum for this H chunk
    offs_h = h0 + tl.arange(0, BLOCK_H)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for j in range(0, NVB + 1):
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(
                tl.float32
            )
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
        acc += p_j * v
    tl.store(
        out_ptr + pid_t * stride_om + offs_h,
        acc.to(out_ptr.dtype.element_ty),
    )


def _mix_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:
    """Triton score + combine pair: returns the pre-norm mixture."""
    T, H = prefix_sum.shape
    cw = get_cw(score_proj, score_norm)
    n_h_blocks = H // _BLOCK_H

    # Step 1: score each row (2D grid, full row-parallelism)
    scores = torch.empty((T, _MAX_ROWS), dtype=torch.float32, device=prefix_sum.device)
    _score_kernel[(T, nvb + 1)](
        prefix_sum,
        bank,
        cw,
        scores,
        nvb,
        score_norm.variance_epsilon,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=H,
        BLOCK_H=_BLOCK_H,
        num_warps=8,
    )

    # Step 2: softmax + weighted sum (2D grid, full H-parallelism)
    out = torch.empty_like(prefix_sum)
    _combine_kernel[(T, n_h_blocks)](
        prefix_sum,
        bank,
        scores,
        out,
        nvb,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )
    return out


def _aggregate_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
) -> torch.Tensor:
    # Step 3: standard RMSNorm (sglang's optimized kernel)
    return out_norm(_mix_fused(prefix_sum, bank, nvb, score_proj, score_norm))


def _aggregate_hip(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: Optional[RMSNorm],
    write_bank_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single ROCm Triton kernel: the bank stays in registers so scoring and
    mixing share one read, and the pending residual add, the bank snapshot and
    the output RMSNorm all fold into the same launch. out_norm None returns the
    pre-norm mixture instead. Returns (result, prefix)."""
    from sglang.kernels.ops.kimi_k3.attn_res_hip import attn_res_hip

    cw = get_cw(score_proj, score_norm)
    prefix = prefix_sum if addend is None else torch.empty_like(prefix_sum)
    out = torch.empty_like(prefix_sum)
    attn_res_hip(
        prefix_sum,
        bank,
        cw,
        out_norm.weight if out_norm is not None else None,
        out,
        nvb,
        score_norm.variance_epsilon,
        out_norm.variance_epsilon if out_norm is not None else 0.0,
        addend=addend,
        prefix_out=prefix,
        write_prefix=write_bank_row,
    )
    return out, prefix


def aggregate_stream_torch(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:
    """Eager reference for aggregate_stream (materializes [T, R, H])."""
    if nvb == 0:
        return prefix_sum
    T, H = prefix_sum.shape
    # rows = [bank[0..nvb-1], prefix_sum]  shape [T, nvb+1, H]
    rows = torch.cat([bank[:, :nvb, :], prefix_sum.unsqueeze(1)], dim=1)
    R = nvb + 1
    normed = score_norm(rows.reshape(T * R, H))
    scores = score_proj(normed)[0].reshape(T, R)
    probs = torch.softmax(scores.float(), dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return mixed.to(prefix_sum.dtype)


def aggregate_stream(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:
    """Pre-norm aggregated stream value (softmax mixture, no output norm):
    the K3 analogue of the residual stream, for dspark aux capture -- the
    raw wire only carries the current block's running prefix."""
    if nvb == 0:
        return prefix_sum
    if _use_hip_fused(prefix_sum.shape[1], nvb):
        return _aggregate_hip(
            prefix_sum, None, bank, nvb, score_proj, score_norm, None
        )[0]
    if prefix_sum.shape[1] % _BLOCK_H != 0:
        return aggregate_stream_torch(prefix_sum, bank, nvb, score_proj, score_norm)
    return _mix_fused(prefix_sum, bank, nvb, score_proj, score_norm)


def _aggregate_fused_add(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    write_bank_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregation point with a pending upstream residual add: materialize
    prefix = prefix_a + prefix_b, then aggregate. Returns (normed, prefix).
    write_bank_row rides _aggregate (fast path only)."""
    if _use_hip_fused(prefix_a.shape[1], nvb):
        # The hip kernel reads the prefix row anyway, so the add folds into it.
        return _aggregate_hip(
            prefix_a,
            prefix_b,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )
    prefix = prefix_a + prefix_b
    return (
        _aggregate(
            prefix,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        ),
        prefix,
    )


def _aggregate(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    write_bank_row: bool = False,
) -> torch.Tensor:
    """Single aggregation point: score → softmax → mix → norm.

    Caller handles nvb == 0 (layer 0 attn side: just out_norm(prefix_sum)).
    write_bank_row is fast-path only (in-kernel snapshot of the prefix row
    into bank[:, nvb, :]); the triton path keeps the standalone .write() copy —
    the caller (AttnResidual.forward) owns that fallback.
    """
    if _use_fast(prefix_sum.shape[1]):
        return _aggregate_fast(
            prefix_sum,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )
    if _use_hip_fused(prefix_sum.shape[1], nvb):
        return _aggregate_hip(
            prefix_sum,
            None,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )[0]
    assert not write_bank_row, "fused bank write is fast-path only"
    return _aggregate_fused(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)


class AttnResidual:
    """Snapshot bank + aggregation of one K3 attention-residual stream,
    backed by the capability-dispatched kernels above.

    One instance lives for one model forward pass.
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        block_num: int,
        block_residual: Optional[torch.Tensor] = None,
    ) -> None:
        num_tokens, hidden_size = hidden_states.shape
        # Frozen snapshot rows [T, NB, H]; raw tensor for PP transfer and the
        # legacy kernel path.
        self.block_residual = hidden_states.new_empty(
            (num_tokens, block_num, hidden_size)
        )
        self.num_valid_blocks = 0
        if block_residual is not None:  # inherited from the previous PP rank
            self.num_valid_blocks = block_residual.size(1)
            self.block_residual[:, : self.num_valid_blocks, :].copy_(block_residual)

    def write(self, prefix_sum: torch.Tensor, rows: Optional[slice] = None) -> None:
        """Snapshot the pre-attention prefix into the next bank row.

        Under SP attention-residual carry each rank owns a disjoint token
        slice, so only that slice is written and subsequently read locally.
        """
        bank = self.block_residual if rows is None else self.block_residual[rows]
        bank[:, self.num_valid_blocks, :].copy_(prefix_sum)
        self.num_valid_blocks += 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        score_proj: ReplicatedLinear,
        score_norm: RMSNorm,
        out_norm: RMSNorm,
        rows: Optional[slice] = None,
        write: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate; with write=True also snapshot the aggregated prefix
        (the second return value) into the next bank row — fused into the
        fast kernel (the row streams through its score pass anyway), a
        standalone .write() copy on every other path."""
        nvb = self.num_valid_blocks
        # Layer 0 attention side: nothing banked yet
        if nvb == 0:
            assert prefix_sum is None
            if write:
                self.write(hidden_states, rows)
            return out_norm(hidden_states), hidden_states

        # SP-MoE: the caller holds only its token shard; align the banked
        # residual rows to it (dim-0 slice of a contiguous buffer stays
        # contiguous for the jit kernels).
        block_residual = (
            self.block_residual if rows is None else self.block_residual[rows]
        )

        fused_write = write and (
            _use_fast(hidden_states.shape[1])
            or _use_hip_fused(hidden_states.shape[1], nvb)
        )
        if prefix_sum is None:
            # hidden_states already is the whole head (PP entry or a
            # block-boundary restart).
            normed = _aggregate(
                hidden_states,
                block_residual,
                nvb,
                score_proj,
                score_norm,
                out_norm,
                write_bank_row=fused_write,
            )
            prefix = hidden_states
        else:
            # Pending add: materialize the prefix, then aggregate.
            normed, prefix = _aggregate_fused_add(
                prefix_sum,
                hidden_states,
                block_residual,
                nvb,
                score_proj,
                score_norm,
                out_norm,
                write_bank_row=fused_write,
            )
        if fused_write:
            self.num_valid_blocks += 1  # row nvb written in-kernel
        elif write:
            self.write(prefix, rows)
        return normed, prefix

    def forward_sp_all_gather(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        score_proj: ReplicatedLinear,
        score_norm: RMSNorm,
        out_norm: RMSNorm,
        rows: slice,
        write: bool = False,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Fuse a local aggregation point and the following row all-gather."""
        nvb = self.num_valid_blocks
        if nvb == 0:
            return None
        if prefix_sum is not None and prefix_sum.shape != hidden_states.shape:
            prefix_sum = prefix_sum[rows]
        prefix = hidden_states if prefix_sum is None else prefix_sum.add(hidden_states)
        bank = self.block_residual[rows]
        cw = get_cw(score_proj, score_norm, dtype=torch.bfloat16)
        assert score_norm.variance_epsilon == out_norm.variance_epsilon
        from sglang.srt.layers import k3_sp_collective

        normed = k3_sp_collective.attn_res_all_gather(
            prefix,
            bank,
            cw,
            out_norm.weight,
            nvb,
            score_norm.variance_epsilon,
            write_prefix=write,
        )
        if normed is None:
            return None
        if write:
            self.num_valid_blocks += 1
        return normed, prefix

    def forward_sp_reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        score_proj: ReplicatedLinear,
        score_norm: RMSNorm,
        out_norm: RMSNorm,
        rows: slice,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Fuse o_proj RS, the pending local prefix add, and aggregation."""
        nvb = self.num_valid_blocks
        if nvb == 0:
            return None
        local_tokens = rows.stop - rows.start
        residual = prefix_sum
        if residual is not None and residual.shape[0] != local_tokens:
            residual = residual[rows]
        bank = self.block_residual[rows]
        cw = get_cw(score_proj, score_norm, dtype=torch.bfloat16)
        assert score_norm.variance_epsilon == out_norm.variance_epsilon
        from sglang.srt.layers import k3_sp_collective

        return k3_sp_collective.reduce_scatter_attn_res(
            hidden_states,
            residual,
            bank,
            cw,
            out_norm.weight,
            nvb,
            score_norm.variance_epsilon,
        )
