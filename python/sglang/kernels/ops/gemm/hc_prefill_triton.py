"""Prefill-sized gated-residual HC mix and combine (per-branch norm) for gfx950.

Coverage, as tested: gfx950 (the launch configs and the 16x16 MFMA were measured there),
BF16, HC 4, hidden size a multiple of 512, low-rank a multiple of 64 (Qwen3.8-Flash-Next:
hidden 2560, low-rank 320), any row count. The normed residual is never materialized:

* ``hc_prefill_mix``: the down and inject projections run as one batched GEMM over the
  HC branches of the raw residual ``h``, against weights with the Gemma norm's
  ``(1 + w)`` folded in (``fold_hc_prefill_weights``). A finish kernel scales each
  branch's partial by that branch's ``rstd``, applies SiLU(. / HC) and returns the
  inject logits. The up kernel rebuilds the gate operand as ``h * rstd_b * (1 + w)``
  and runs the up projection, the sigmoid and the branch mean.
* ``hc_prefill_combine``: ``h + y * 2 sigmoid(g / HC)`` from the inject logits.

M is a runtime argument that Triton does not specialize: a new row count compiles
nothing, and every launch can be captured in a CUDA graph.

Numerics differ from the unfused path only by rounding points: ``(1 + w)`` is rounded
into the BF16 down and inject weights, each branch's down partial is rounded to BF16
before its ``rstd`` scaling, the normed residual is never rounded to BF16, and the up
projection stays FP32 up to the sigmoid.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.arch import current_arch


@triton.jit
def _sigmoid(v):
    # gfx950: v_exp_f32 + v_rcp_f32; 1 / (1 + tl.exp(-v)) lowers to the IEEE division
    # sequence (24 VALU instructions instead of 7).
    return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-v))


@triton.jit(do_not_specialize=["M"])
def _hc_prefill_sumsq_kernel(
    h_ptr,
    sumsq_ptr,
    M,
    HS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """sumsq[M, HC]: per-branch sums of squares of h[M, HC * HS].

    Grid (cdiv(M, BLOCK_M), HC)."""
    K: tl.constexpr = HC * HS
    branch = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows64 = rows.to(tl.int64)
    acc = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
    for k0 in range(0, HS, BLOCK_K):
        x = tl.load(
            h_ptr
            + rows64[:, None] * K
            + branch * HS
            + k0
            + tl.arange(0, BLOCK_K)[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += x * x
    tl.store(sumsq_ptr + rows64 * HC + branch, tl.sum(acc, axis=1), mask=row_mask)


@triton.jit(do_not_specialize=["M"])
def _hc_prefill_finish_kernel(
    partial_ptr,
    sumsq_ptr,
    t_ptr,
    logits_ptr,
    rstd_ptr,
    M,
    eps,
    HS: tl.constexpr,
    HC: tl.constexpr,
    LOWRANK: tl.constexpr,
    PARTIAL_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Per-branch down and inject partials partial[HC, M, PARTIAL_COLS] and sums of
    squares sumsq[M, HC] -> t = SiLU(sum_b rstd_b partial_b / HC) [M, LOWRANK] BF16,
    inject logits sum_b rstd_b partial_b [M, HC] FP32 and rstd [M, HC] FP32.

    Grid (cdiv(M, BLOCK_M), cdiv(LOWRANK + HC, BLOCK_C))."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows64 = rows.to(tl.int64)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    acc = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    for branch in tl.static_range(HC):
        sumsq = tl.load(sumsq_ptr + rows64 * HC + branch, mask=row_mask, other=0.0)
        rstd = tl.math.rsqrt(sumsq / HS + eps)
        if tl.program_id(1) == 0:
            tl.store(rstd_ptr + rows64 * HC + branch, rstd, mask=row_mask)
        partial = tl.load(
            partial_ptr
            + branch * M.to(tl.int64) * PARTIAL_COLS
            + rows64[:, None] * PARTIAL_COLS
            + cols[None, :],
            mask=row_mask[:, None] & (cols < LOWRANK + HC)[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += partial * rstd[:, None]
    v = acc / HC
    tl.store(
        t_ptr + rows64[:, None] * LOWRANK + cols[None, :],
        (v * _sigmoid(v)).to(tl.bfloat16),
        mask=row_mask[:, None] & (cols < LOWRANK)[None, :],
    )
    tl.store(
        logits_ptr + rows64[:, None] * HC + (cols - LOWRANK)[None, :],
        acc,
        mask=row_mask[:, None] & ((cols >= LOWRANK) & (cols < LOWRANK + HC))[None, :],
    )


@triton.jit(do_not_specialize=["M"])
def _hc_prefill_up_kernel(
    t_ptr,
    up_ptr,
    h_ptr,
    rstd_ptr,
    norm_weight_ptr,
    out_ptr,
    M,
    HS: tl.constexpr,
    HC: tl.constexpr,
    LOWRANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """out[m, j] = mean_b sigmoid(t[m] . up[n]) * h[m, n] * rstd[m, b] * (1 + w[n]),
    n = b * HS + j.

    Grid (cdiv(M, BLOCK_M), HS // BLOCK_J). One [BLOCK_M, HC * BLOCK_J] GEMM tile per
    CTA whose columns are the HC branches' copies of hidden units j; the low-rank
    dimension is the K loop in BLOCK_R chunks, and the branch mean reduces the tile's
    HC column groups.
    """
    K: tl.constexpr = HC * HS
    TILE_N: tl.constexpr = HC * BLOCK_J
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows64 = rows.to(tl.int64)
    pid_j = tl.program_id(1)
    c = tl.arange(0, TILE_N)
    branch = c // BLOCK_J
    n = branch * HS + pid_j * BLOCK_J + c % BLOCK_J
    acc = tl.zeros([BLOCK_M, TILE_N], tl.float32)
    for r0 in range(0, LOWRANK, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        t = tl.load(
            t_ptr + rows64[:, None] * LOWRANK + r[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        w = tl.load(up_ptr + n[:, None] * LOWRANK + r[None, :])
        acc = tl.dot(t, tl.trans(w), acc)
    h = tl.load(
        h_ptr + rows64[:, None] * K + n[None, :], mask=row_mask[:, None], other=0.0
    ).to(tl.float32)
    rstd = tl.load(
        rstd_ptr + rows64[:, None] * HC + branch[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    norm_scale = 1.0 + tl.load(norm_weight_ptr + n).to(tl.float32)
    gated = _sigmoid(acc) * h * rstd * norm_scale[None, :]
    mixed = tl.sum(tl.reshape(gated, [BLOCK_M, HC, BLOCK_J]), axis=1) / HC
    j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    tl.store(
        out_ptr + rows64[:, None] * HS + j[None, :],
        mixed.to(tl.bfloat16),
        mask=row_mask[:, None],
    )


@triton.jit(do_not_specialize=["M"])
def _hc_prefill_combine_kernel(
    h_ptr,
    y_ptr,
    logits_ptr,
    out_ptr,
    M,
    HS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """out[m, b * HS + j] = h[m, b * HS + j] + 2 sigmoid(logits[m, b] / HC) * y[m, j].

    Grid (cdiv(M, BLOCK_M), HS // BLOCK_J); each CTA reads its y tile once for all
    branches."""
    K: tl.constexpr = HC * HS
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows64 = rows.to(tl.int64)
    j = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
    y = tl.load(
        y_ptr + rows64[:, None] * HS + j[None, :], mask=row_mask[:, None], other=0.0
    ).to(tl.float32)
    for branch in tl.static_range(HC):
        logits = tl.load(logits_ptr + rows64 * HC + branch, mask=row_mask, other=0.0)
        gate = 2.0 * _sigmoid(logits / HC)
        offsets = rows64[:, None] * K + branch * HS + j[None, :]
        h = tl.load(h_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        tl.store(
            out_ptr + offsets,
            (h + gate[:, None] * y).to(tl.bfloat16),
            mask=row_mask[:, None],
        )


# Launch configs from per-kernel sweeps on MI355X (gfx950) at HC 4, hidden 2560 and
# low-rank 320.
_SUMSQ_BLOCK_M, _SUMSQ_BLOCK_K = 4, 512
_FINISH_BLOCK_M, _FINISH_BLOCK_C = 32, 64
_UP_BLOCK_J, _UP_BLOCK_R = 32, 64
_COMBINE_BLOCK_M, _COMBINE_BLOCK_J = 4, 512


def _up_config(rows: int) -> dict:
    # matrix_instr_nonkdim=16 selects the 16x16 MFMA.
    if rows <= 1024:
        return dict(BLOCK_M=64, num_warps=4, num_stages=3, matrix_instr_nonkdim=16)
    return dict(BLOCK_M=128, num_warps=8, num_stages=2, matrix_instr_nonkdim=16)


def fold_hc_prefill_weights(
    down: torch.Tensor, inject: torch.Tensor, norm_weight: torch.Tensor
) -> torch.Tensor:
    """The down [LOWRANK, K] and inject [HC, K] weights times the Gemma norm's
    ``(1 + w)``, stacked and zero-padded to a multiple of 16 rows, in BF16: the
    projections then read the raw residual. Build it after weight loading."""
    scale = 1.0 + norm_weight.float()
    folded = torch.cat([down.float() * scale, inject.float() * scale])
    return F.pad(folded, (0, 0, 0, -folded.shape[0] % 16)).to(torch.bfloat16)


def hc_prefill_weights_supported(down: torch.Tensor, up: torch.Tensor, hc: int) -> bool:
    """Architecture and weight-shape half of the check.

    None of it changes after the weights are loaded, so callers evaluate it once per
    layer. ``down`` is [LOWRANK, HC * HIDDEN] and ``up`` [HC * HIDDEN, LOWRANK].
    """
    hip = current_arch().hip
    lowrank, k = down.shape
    return (
        hip is not None
        and hip.gcn_arch == "gfx950"
        and hc == 4
        and (k // hc) % _SUMSQ_BLOCK_K == 0
        and (k // hc) % _COMBINE_BLOCK_J == 0
        and lowrank % _UP_BLOCK_R == 0
        and up.dtype == torch.bfloat16
    )


def hc_prefill_supported(hidden: torch.Tensor) -> bool:
    """Per-call half: the residual's dtype and layout."""
    return (
        hidden.dtype == torch.bfloat16 and hidden.ndim == 2 and hidden.is_contiguous()
    )


def hc_prefill_mix(
    hidden: torch.Tensor,
    folded_weight: torch.Tensor,
    up: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The mix of one HC site on the residual ``hidden`` [M, HC * HS] BF16.

    ``folded_weight`` is ``fold_hc_prefill_weights`` of the site's down and inject
    weights and its norm weight; ``up`` [HC * HS, LOWRANK] is the up weight. Both
    halves of the support check have accepted the weights and ``hidden``. Returns the
    mixed input [M, HS] BF16 and the inject logits [M, HC] FP32.
    """
    rows, k = hidden.shape
    hs = k // hc
    lowrank = up.shape[1]
    device = hidden.device
    sumsq = torch.empty((rows, hc), device=device, dtype=torch.float32)
    _hc_prefill_sumsq_kernel[(triton.cdiv(rows, _SUMSQ_BLOCK_M), hc)](
        hidden,
        sumsq,
        rows,
        HS=hs,
        HC=hc,
        BLOCK_M=_SUMSQ_BLOCK_M,
        BLOCK_K=_SUMSQ_BLOCK_K,
        num_warps=4,
    )
    # Branch-batched over strided views of the residual and the weights; no copies.
    partial = torch.bmm(
        hidden.view(rows, hc, hs).transpose(0, 1),
        folded_weight.view(-1, hc, hs).permute(1, 2, 0),
    )
    t = torch.empty((rows, lowrank), device=device, dtype=torch.bfloat16)
    logits = torch.empty((rows, hc), device=device, dtype=torch.float32)
    rstd = torch.empty((rows, hc), device=device, dtype=torch.float32)
    _hc_prefill_finish_kernel[
        (
            triton.cdiv(rows, _FINISH_BLOCK_M),
            triton.cdiv(lowrank + hc, _FINISH_BLOCK_C),
        )
    ](
        partial,
        sumsq,
        t,
        logits,
        rstd,
        rows,
        eps,
        HS=hs,
        HC=hc,
        LOWRANK=lowrank,
        PARTIAL_COLS=folded_weight.shape[0],
        BLOCK_M=_FINISH_BLOCK_M,
        BLOCK_C=_FINISH_BLOCK_C,
        num_warps=4,
    )
    mixed = torch.empty((rows, hs), device=device, dtype=torch.bfloat16)
    config = _up_config(rows)
    _hc_prefill_up_kernel[(triton.cdiv(rows, config["BLOCK_M"]), hs // _UP_BLOCK_J)](
        t,
        up,
        hidden,
        rstd,
        norm_weight,
        mixed,
        rows,
        HS=hs,
        HC=hc,
        LOWRANK=lowrank,
        BLOCK_J=_UP_BLOCK_J,
        BLOCK_R=_UP_BLOCK_R,
        **config,
    )
    return mixed, logits


def hc_prefill_combine(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    inject_logits: torch.Tensor,
    hc: int,
) -> torch.Tensor:
    """``hidden`` [M, HC * HS] BF16 plus ``block_output`` [M, HS] times
    2 * sigmoid(``inject_logits`` [M, HC] FP32 / HC), per branch, in BF16.

    ``inject_logits`` is what ``hc_prefill_mix`` returned for ``hidden``. All three
    must be contiguous.
    """
    rows, k = hidden.shape
    hs = k // hc
    if (
        hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or block_output.shape != (rows, hs)
        or not block_output.is_contiguous()
        or inject_logits.shape != (rows, hc)
        or inject_logits.dtype != torch.float32
        or not inject_logits.is_contiguous()
    ):
        raise ValueError("unsupported HC prefill combine operands")
    out = torch.empty_like(hidden)
    _hc_prefill_combine_kernel[
        (triton.cdiv(rows, _COMBINE_BLOCK_M), hs // _COMBINE_BLOCK_J)
    ](
        hidden,
        block_output,
        inject_logits,
        out,
        rows,
        HS=hs,
        HC=hc,
        BLOCK_M=_COMBINE_BLOCK_M,
        BLOCK_J=_COMBINE_BLOCK_J,
        num_warps=8,
    )
    return out
