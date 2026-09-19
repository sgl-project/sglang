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
#           SM. Taken on SM100+ except SM12x with H=7168.
#   sm8x  — one Triton CTA per token, fusing an optional prefix add, online
#           softmax aggregation, output RMSNorm, and optional bank write.
#           Taken on the PPU SM80/SM89-compatible path.
#   hip   — single Triton kernel, everything in one launch; taken on ROCm
#           within its register budget.
#   fused — Triton 2-kernel pipeline with full H-parallelism; the fallback
#           everywhere the specialized kernels do not apply.
# aggregate_stream_torch is the eager reference (tests and the
# H % _BLOCK_H != 0 shape fallback of aggregate_stream).

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import is_hip, is_npu, is_ppu

_BLOCK_H: int = 1024  # H = 7168 = 7 x 1024
_MAX_ROWS: int = 16  # next_pow2(8 + 1), K3 has <= 8 snapshots

_FAST_SUPPORTED = None
_HIP_SHAPE_GATE = None
_SM8X_FUSED_SUPPORTED = None


def _supports_attn_res_tma(capability: tuple[int, int]) -> bool:
    """Return whether the device is eligible for the TMA fast path."""
    major, _ = capability
    return major >= 10 and major != 12


def _use_fast(hidden_size: int) -> bool:
    """The TMA kernel needs SM100+ except SM12x (tcgen05, cp.async.bulk)
    and its H=7168 template; everything else takes the triton pipeline."""
    global _FAST_SUPPORTED
    if is_npu():
        return False
    if _FAST_SUPPORTED is None:
        _FAST_SUPPORTED = _supports_attn_res_tma(torch.cuda.get_device_capability())
    return _FAST_SUPPORTED and hidden_size == 7168


def _use_hip_fused(hidden_size: int, nvb: int) -> bool:
    """This gate picks the single-kernel ROCm Triton kernel instead of the
    2-kernel pipeline."""
    if not is_hip():
        return False
    global _HIP_SHAPE_GATE
    if _HIP_SHAPE_GATE is None:
        from sglang.kernels.ops.attention.attn_res_hip import supports_attn_res_hip

        _HIP_SHAPE_GATE = supports_attn_res_hip
    return _HIP_SHAPE_GATE(hidden_size, nvb)


def _use_sm8x_fused(hidden_size: int) -> bool:
    """The single-CTA Triton kernel is tuned for PPU K3's SM8x path only."""
    global _SM8X_FUSED_SUPPORTED
    if _SM8X_FUSED_SUPPORTED is None:
        capability = torch.cuda.get_device_capability()
        _SM8X_FUSED_SUPPORTED = (
            is_ppu() and torch.version.hip is None and capability in ((8, 0), (8, 9))
        )
    return _SM8X_FUSED_SUPPORTED and hidden_size == 7168


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
    from sglang.kernels.ops.attention.attn_res import attn_res_fused_tma

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


# ---- SM80/SM89 fused aggregation --------------------------------------------


@triton.jit
def _aggregate_sm8x_kernel(
    prefix_ptr,  # [T, H]
    delta_ptr,  # [T, H] when HAS_DELTA
    prefix_out_ptr,  # [T, H], receives prefix + delta when HAS_DELTA
    bank_ptr,  # [T, NB_total, H]
    cw_ptr,  # [H], fp32
    out_weight_ptr,  # [H]
    out_ptr,  # [T, H]
    stride_pm: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_om: tl.constexpr,
    NVB: tl.constexpr,
    EPS: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BANK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One CTA per token for the SM80/SM89 K3 attention-residual path.

    The kernel deliberately rounds the pending prefix add back to the source
    dtype. This is the value consumed by the old eager add plus fallback and
    must also be the value retained by AttnResidual as its running prefix.
    """
    pid_t = tl.program_id(0).to(tl.int64)
    offs_h = tl.max_contiguous(tl.arange(0, BLOCK_H), BLOCK_H)
    mask_h = offs_h < H

    updated_prefix = tl.load(
        prefix_ptr + pid_t * stride_pm + offs_h, mask=mask_h, other=0.0
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + pid_t * stride_dm + offs_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        updated_prefix = updated_prefix + delta
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_out_ptr + pid_t * stride_pm + offs_h, updated_prefix, mask=mask_h
        )

    if WRITE_BANK:
        tl.store(
            bank_ptr + pid_t * stride_bm + NVB * stride_bb + offs_h,
            updated_prefix,
            mask=mask_h,
        )

    # A block write with NVB == 0 is the first K3 attention-residual point:
    # the sole source has softmax weight one, so it needs no score pass.
    if NVB == 0:
        mixed = updated_prefix
    else:
        # Do not retain the 8192-wide prefix vector over the source loop.
        # Selecting source pointers lets Triton reload it in the prefix tile,
        # avoiding the severe register pressure observed on SM89/PPU.
        if HAS_DELTA:
            tl.debug_barrier()
            prefix_source_ptr = prefix_out_ptr
        else:
            prefix_source_ptr = prefix_ptr
        cw = tl.load(cw_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        max_score = tl.full((), -float("inf"), tl.float32)
        denom = tl.zeros((), tl.float32)
        mixed = tl.zeros([BLOCK_H], tl.float32)

        # Process bank rows plus current prefix in small tiles. Online softmax
        # avoids materializing scores and preserves a single kernel launch.
        for row0 in tl.static_range(0, NVB + 1, BLOCK_ROWS):
            offs_r = row0 + tl.arange(0, BLOCK_ROWS)
            row_mask = offs_r <= NVB
            is_prefix = offs_r == NVB
            bank_ptrs = (
                bank_ptr
                + pid_t * stride_bm
                + offs_r[:, None] * stride_bb
                + offs_h[None, :]
            )
            prefix_ptrs = (
                prefix_source_ptr
                + pid_t * stride_pm
                + offs_r[:, None] * 0
                + offs_h[None, :]
            )
            values = tl.load(
                tl.where(is_prefix[:, None], prefix_ptrs, bank_ptrs),
                mask=row_mask[:, None] & mask_h[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(values * values, axis=1) * (1.0 / H) + EPS)
            scores = tl.sum(values * cw[None, :], axis=1) * inv_rms
            scores = tl.where(row_mask, scores, -float("inf"))

            new_max = tl.maximum(max_score, tl.max(scores, axis=0))
            old_scale = tl.exp(max_score - new_max)
            scales = tl.exp(scores - new_max)
            denom = denom * old_scale + tl.sum(scales, axis=0)
            mixed = mixed * old_scale + tl.sum(scales[:, None] * values, axis=0)
            max_score = new_max

        mixed = mixed / denom

    out_inv_rms = tl.rsqrt(tl.sum(mixed * mixed, axis=0) * (1.0 / H) + EPS)
    out_weight = tl.load(out_weight_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + pid_t * stride_om + offs_h,
        mixed * out_inv_rms * out_weight,
        mask=mask_h,
    )


def _aggregate_sm8x(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    delta: Optional[torch.Tensor] = None,
    write_bank_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SM80/SM89 aggregate returning ``(normed, prefix)``.

    The optional pending addition is rounded and written to a newly allocated
    prefix buffer by this kernel. This preserves input aliasing while avoiding
    a separate prefix-add or copy launch.
    """
    assert score_norm.variance_epsilon == out_norm.variance_epsilon
    assert prefix_sum.is_contiguous() and bank.is_contiguous()
    assert delta is None or delta.is_contiguous()
    T, H = prefix_sum.shape
    out = torch.empty_like(prefix_sum)
    prefix = torch.empty_like(prefix_sum) if delta is not None else prefix_sum
    block_rows = 1 if T >= 256 or nvb <= 1 else 4
    _aggregate_sm8x_kernel[(T,)](
        prefix_sum,
        prefix_sum if delta is None else delta,
        prefix,
        bank,
        get_cw(score_proj, score_norm),
        out_norm.weight,
        out,
        prefix_sum.stride(0),
        0 if delta is None else delta.stride(0),
        bank.stride(0),
        bank.stride(1),
        out.stride(0),
        NVB=nvb,
        EPS=score_norm.variance_epsilon,
        HAS_DELTA=delta is not None,
        WRITE_BANK=write_bank_row,
        BLOCK_ROWS=block_rows,
        H=H,
        BLOCK_H=triton.next_power_of_2(H),
        num_warps=4 if block_rows == 1 else 8,
        num_stages=2,
    )
    return out, prefix


# ---- Kernel 1: per-row scoring (2D grid [T, NVB+1]) -------------------------
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
    pid_t = tl.program_id(0).to(tl.int64)
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
    pid_t = tl.program_id(0).to(tl.int64)
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
    if T == 0:
        return prefix_sum
    cw = get_cw(score_proj, score_norm)
    if is_npu():
        from sgl_kernel_npu.kimi_k3.attn_residual import mix_fused

        return mix_fused(
            prefix_sum,
            bank,
            nvb,
            cw,
            score_norm.variance_epsilon,
        )
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
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

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

    The caller handles non-write nvb == 0 (layer 0: just out_norm(prefix_sum)).
    Specialized paths can snapshot ``prefix_sum`` into ``bank[:, nvb, :]`` in
    the same launch; the SM8x path also handles an initial nvb == 0 snapshot.
    The 2-kernel Triton fallback keeps the standalone copy.
    """
    if prefix_sum.shape[0] == 0:
        return prefix_sum
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
    if _use_sm8x_fused(prefix_sum.shape[1]):
        return _aggregate_sm8x(
            prefix_sum,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )[0]
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
        specialized kernel (the row streams through its score pass anyway), a
        standalone .write() copy on every other path."""
        nvb = self.num_valid_blocks
        # Layer 0 attention side: nothing banked yet
        if nvb == 0:
            assert prefix_sum is None
            if write and _use_sm8x_fused(hidden_states.shape[1]):
                bank = (
                    self.block_residual if rows is None else self.block_residual[rows]
                )
                normed, prefix = _aggregate_sm8x(
                    hidden_states,
                    bank,
                    nvb,
                    score_proj,
                    score_norm,
                    out_norm,
                    write_bank_row=True,
                )
                self.num_valid_blocks += 1  # row 0 written in-kernel
                return normed, prefix
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
            or _use_sm8x_fused(hidden_states.shape[1])
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
        elif _use_sm8x_fused(hidden_states.shape[1]):
            # The fused kernel materializes the running prefix into its own
            # output buffer, leaving a caller-owned prefix_sum untouched.
            normed, prefix = _aggregate_sm8x(
                prefix_sum,
                block_residual,
                nvb,
                score_proj,
                score_norm,
                out_norm,
                delta=hidden_states,
                write_bank_row=fused_write,
            )
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
        from sglang.srt.layers.communication import k3_sp_collective

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
        from sglang.srt.layers.communication import k3_sp_collective

        return k3_sp_collective.reduce_scatter_attn_res(
            hidden_states,
            residual,
            bank,
            cw,
            out_norm.weight,
            nvb,
            score_norm.variance_epsilon,
        )
