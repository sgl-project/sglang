"""Prefill-sized gated-residual HC mix and combine (per-branch norm) for gfx950.

Coverage, as tested: gfx950 (the launch configs and the 16x16 MFMA were measured there),
BF16, HC 4, hidden size a multiple of 512, low-rank a multiple of 64 (Qwen3.8-Flash-Next:
hidden 2560, low-rank 320), any row count. The normed residual is never materialized:

* ``hc_prefill_mix``: the down and inject projections run as one batched GEMM over the
  HC branches of the raw residual ``h``, against weights with the Gemma norm's
  ``(1 + w)`` folded in (``fold_hc_prefill_weights``): ``torch.bmm``, or at most
  ``_SPLIT_K_MAX_ROWS`` rows a split-K Triton kernel that cuts K into
  ``_SPLITS_PER_BRANCH`` slices per branch. A finish kernel sums a branch's partials,
  scales them by that branch's ``rstd``, applies SiLU(. / HC) and returns the inject
  logits. The up kernel rebuilds the gate operand as ``h * rstd_b * (1 + w)`` and runs
  the up projection, the sigmoid and the branch mean.
* ``hc_prefill_combine``: ``h + y * 2 sigmoid(g / HC)`` from the inject logits.

The row count is never a compile-time constant (Triton does not specialize ``M``, and
the combine's grid carries it): a new row count compiles nothing, and every launch can
be captured in a CUDA graph.

Numerics differ from the unfused path only by rounding points: ``(1 + w)`` is rounded
into the BF16 down and inject weights, each branch's down partial (each K slice's, on
the split-K path) is rounded to BF16 before its ``rstd`` scaling, the normed residual
is never rounded to BF16, and the up projection stays FP32 up to the sigmoid.
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


@triton.jit
def _xcd_order(program_id, num_tiles, NUM_GROUPS: tl.constexpr, NUM_XCDS: tl.constexpr):
    """(group, tile) of ``program_id`` in a 1-D grid of NUM_GROUPS * num_tiles programs.
    The programs of a group share program_id % NUM_XCDS: workgroups go to the XCDs
    round-robin, so a group stays on one XCD and its L2."""
    tl.static_assert(
        NUM_GROUPS % NUM_XCDS == 0, "a group count that is not a multiple of NUM_XCDS"
    )
    quotient = program_id // NUM_XCDS
    group = program_id % NUM_XCDS + NUM_XCDS * (quotient // num_tiles)
    return group, quotient % num_tiles


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
def _hc_prefill_down_kernel(
    hidden_ptr,
    weight_ptr,
    partial_ptr,
    M,
    K: tl.constexpr,
    PARTIAL_COLS: tl.constexpr,
    SPLITS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """partial[s, m, n] = sum over K slice s of hidden[m, k] * weight[n, k], in BF16.

    1-D grid of SPLITS * cdiv(M, BLOCK_M) * cdiv(PARTIAL_COLS, BLOCK_N) programs."""
    tl.static_assert(K % (SPLITS * BLOCK_K) == 0, "the K slices do not tile K")
    SLICE_K: tl.constexpr = K // SPLITS
    NUM_COL_TILES: tl.constexpr = (PARTIAL_COLS + BLOCK_N - 1) // BLOCK_N
    num_row_tiles = tl.cdiv(M, BLOCK_M)
    split, tile = _xcd_order(
        tl.program_id(0), num_row_tiles * NUM_COL_TILES, SPLITS, NUM_XCDS
    )
    rows = tile % num_row_tiles * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows64 = rows.to(tl.int64)
    cols = tile // num_row_tiles * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < PARTIAL_COLS
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, SLICE_K, BLOCK_K):
        k = split * SLICE_K + k0 + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_ptr + rows64[:, None] * K + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + cols[:, None] * K + k[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(hidden, tl.trans(weight), acc)
    tl.store(
        partial_ptr
        + (split.to(tl.int64) * M + rows64)[:, None] * PARTIAL_COLS
        + cols[None, :],
        acc.to(tl.bfloat16),
        mask=row_mask[:, None] & col_mask[None, :],
    )


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
    SPLITS_PER_BRANCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Down and inject partials partial[HC * SPLITS_PER_BRANCH, M, PARTIAL_COLS] (branch
    b's slices start at b * SPLITS_PER_BRANCH) and sums of squares sumsq[M, HC]
    -> t = SiLU(sum_b rstd_b partial_b / HC) [M, LOWRANK] BF16, inject logits
    sum_b rstd_b partial_b [M, HC] FP32 and rstd [M, HC] FP32.

    Grid (cdiv(M, BLOCK_M), cdiv(LOWRANK + HC, BLOCK_C))."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows64 = rows.to(tl.int64)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = row_mask[:, None] & (cols < LOWRANK + HC)[None, :]
    branches = tl.arange(0, HC)
    rstds = tl.zeros([BLOCK_M, HC], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    for branch in tl.static_range(HC):
        sumsq = tl.load(sumsq_ptr + rows64 * HC + branch, mask=row_mask, other=0.0)
        rstd = tl.math.rsqrt(sumsq / HS + eps)
        rstds = tl.where(branches[None, :] == branch, rstd[:, None], rstds)
        branch_sum = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
        for split in tl.static_range(SPLITS_PER_BRANCH):
            slice_index = branch * SPLITS_PER_BRANCH + split
            branch_sum += tl.load(
                partial_ptr
                + (slice_index * M.to(tl.int64) + rows64)[:, None] * PARTIAL_COLS
                + cols[None, :],
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
        acc += branch_sum * rstd[:, None]
    # After the loop: a store inside it makes the compiled kernel wait for each
    # branch's loads before it issues the next branch's.
    if tl.program_id(1) == 0:
        tl.store(
            rstd_ptr + rows64[:, None] * HC + branches[None, :],
            rstds,
            mask=row_mask[:, None],
        )
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
    NUM_XCDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """out[m, j] = mean_b sigmoid(t[m] . up[n]) * h[m, n] * rstd[m, b] * (1 + w[n]),
    n = b * HS + j.

    Grid (cdiv(M, BLOCK_M), HS // BLOCK_J), or with NUM_XCDS > 0 one dimension of
    cdiv(M, BLOCK_M) * (HS // BLOCK_J) programs, where the row tiles of a column block
    share program_id % NUM_XCDS. One
    [BLOCK_M, HC * BLOCK_J] GEMM tile per CTA whose columns are the HC branches' copies
    of hidden units j; the low-rank dimension is the K loop in BLOCK_R chunks, and the
    branch mean reduces the tile's HC column groups.
    """
    K: tl.constexpr = HC * HS
    TILE_N: tl.constexpr = HC * BLOCK_J
    if NUM_XCDS == 0:
        pid_m, pid_j = tl.program_id(0), tl.program_id(1)
    else:
        pid_j, pid_m = _xcd_order(
            tl.program_id(0), tl.cdiv(M, BLOCK_M), HS // BLOCK_J, NUM_XCDS
        )
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows64 = rows.to(tl.int64)
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


@triton.jit
def _hc_prefill_combine_kernel(
    h_ptr,
    y_ptr,
    logits_ptr,
    out_ptr,
    HS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """out[m, b * HS + j] = h[m, b * HS + j] + 2 sigmoid(logits[m, b] / HC) * y[m, j].

    Grid (M, HS // BLOCK_J): a program takes one row and BLOCK_J hidden units for all
    HC branches as one [HC, BLOCK_J] tile, reads y once, and loads everything before it
    stores."""
    K: tl.constexpr = HC * HS
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
    branches = tl.arange(0, HC)
    y = tl.load(y_ptr + row * HS + j).to(tl.float32)
    logits = tl.load(logits_ptr + row * HC + branches)
    offsets = row * K + branches[:, None] * HS + j[None, :]
    h = tl.load(h_ptr + offsets).to(tl.float32)
    gate = 2.0 * _sigmoid(logits / HC)
    tl.store(out_ptr + offsets, (h + gate[:, None] * y[None, :]).to(tl.bfloat16))


# Launch configs from per-kernel sweeps on MI355X (gfx950) at HC 4, hidden 2560 and
# low-rank 320.
_SUMSQ_BLOCK_M, _SUMSQ_BLOCK_K = 4, 512
_UP_BLOCK_J, _UP_BLOCK_R = 32, 64
_COMBINE_BLOCK_J = 512
# Split-K path: K splits into _SPLITS_PER_BRANCH slices per branch. The support check
# (HC 4, hidden a multiple of 512) makes the 16 slices a multiple of _NUM_XCDS, a slice
# a multiple of _DOWN_BLOCK_K and the up grid's column blocks a multiple of _NUM_XCDS;
# the kernels static-assert each, so a changed constant fails to compile.
# _SPLIT_K_MAX_ROWS: the largest row count the path was measured against torch.bmm at.
_SPLIT_K_MAX_ROWS = 512
_SPLITS_PER_BRANCH = 4
_NUM_XCDS = 8  # XCDs of an MI355X
_DOWN_BLOCK_M, _DOWN_BLOCK_N, _DOWN_BLOCK_K = 64, 32, 128


def _finish_config(split_k: bool) -> dict:
    if split_k:
        return dict(BLOCK_M=1, BLOCK_C=128, num_warps=2)
    return dict(BLOCK_M=32, BLOCK_C=64, num_warps=4)


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


def hc_prefill_weights_supported(
    down: torch.Tensor, up: torch.Tensor, norm_weight: torch.Tensor, hc: int
) -> bool:
    """Architecture and weight-shape half of the check.

    None of it changes after the weights are loaded, so callers evaluate it once per
    layer. ``down`` is [LOWRANK, HC * HIDDEN], ``up`` [HC * HIDDEN, LOWRANK] and
    ``norm_weight`` [HC * HIDDEN]; the kernels read the last two row-major.
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
        and up.is_contiguous()
        and norm_weight.is_contiguous()
    )


def hc_prefill_supported(hidden: torch.Tensor) -> bool:
    """Per-call half: the residual's dtype and layout."""
    return (
        hidden.dtype == torch.bfloat16 and hidden.ndim == 2 and hidden.is_contiguous()
    )


def _split_k_down(
    hidden: torch.Tensor, folded_weight: torch.Tensor, partial: torch.Tensor
) -> None:
    """Write ``partial`` [SPLITS, M, PARTIAL_COLS] BF16: slice s of K is
    ``hidden[:, s * L : (s + 1) * L] @ folded_weight[:, s * L : (s + 1) * L].T``."""
    splits, rows, partial_cols = partial.shape
    _hc_prefill_down_kernel[
        (
            splits
            * triton.cdiv(rows, _DOWN_BLOCK_M)
            * triton.cdiv(partial_cols, _DOWN_BLOCK_N),
        )
    ](
        hidden,
        folded_weight,
        partial,
        rows,
        K=hidden.shape[1],
        PARTIAL_COLS=partial_cols,
        SPLITS=splits,
        NUM_XCDS=_NUM_XCDS,
        BLOCK_M=_DOWN_BLOCK_M,
        BLOCK_N=_DOWN_BLOCK_N,
        BLOCK_K=_DOWN_BLOCK_K,
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )


def _launch_up(
    t: torch.Tensor,
    up: torch.Tensor,
    hidden: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    mixed: torch.Tensor,
    hc: int,
    xcd_order: bool,
) -> None:
    rows, k = hidden.shape
    config = _up_config(rows)
    row_tiles, column_blocks = (
        triton.cdiv(rows, config["BLOCK_M"]),
        k // hc // _UP_BLOCK_J,
    )
    _hc_prefill_up_kernel[
        (row_tiles * column_blocks,) if xcd_order else (row_tiles, column_blocks)
    ](
        t,
        up,
        hidden,
        rstd,
        norm_weight,
        mixed,
        rows,
        HS=k // hc,
        HC=hc,
        LOWRANK=up.shape[1],
        NUM_XCDS=_NUM_XCDS if xcd_order else 0,
        BLOCK_J=_UP_BLOCK_J,
        BLOCK_R=_UP_BLOCK_R,
        **config,
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
    split_k = rows <= _SPLIT_K_MAX_ROWS
    splits_per_branch = _SPLITS_PER_BRANCH if split_k else 1
    if split_k:
        partial = torch.empty(
            (hc * splits_per_branch, rows, folded_weight.shape[0]),
            device=device,
            dtype=torch.bfloat16,
        )
        _split_k_down(hidden=hidden, folded_weight=folded_weight, partial=partial)
    else:
        # Branch-batched over strided views of the residual and the weights; no copies.
        partial = torch.bmm(
            hidden.view(rows, hc, hs).transpose(0, 1),
            folded_weight.view(-1, hc, hs).permute(1, 2, 0),
        )
    t = torch.empty((rows, lowrank), device=device, dtype=torch.bfloat16)
    logits = torch.empty((rows, hc), device=device, dtype=torch.float32)
    rstd = torch.empty((rows, hc), device=device, dtype=torch.float32)
    finish_config = _finish_config(split_k)
    _hc_prefill_finish_kernel[
        (
            triton.cdiv(rows, finish_config["BLOCK_M"]),
            triton.cdiv(lowrank + hc, finish_config["BLOCK_C"]),
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
        SPLITS_PER_BRANCH=splits_per_branch,
        **finish_config,
    )
    mixed = torch.empty((rows, hs), device=device, dtype=torch.bfloat16)
    _launch_up(
        t=t,
        up=up,
        hidden=hidden,
        rstd=rstd,
        norm_weight=norm_weight,
        mixed=mixed,
        hc=hc,
        xcd_order=split_k,
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
    must be contiguous, and HS a multiple of 512.
    """
    rows, k = hidden.shape
    hs = k // hc
    if (
        hs % _COMBINE_BLOCK_J != 0
        or hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or block_output.shape != (rows, hs)
        or not block_output.is_contiguous()
        or inject_logits.shape != (rows, hc)
        or inject_logits.dtype != torch.float32
        or not inject_logits.is_contiguous()
    ):
        raise ValueError("unsupported HC prefill combine operands")
    out = torch.empty_like(hidden)
    _hc_prefill_combine_kernel[(rows, hs // _COMBINE_BLOCK_J)](
        hidden,
        block_output,
        inject_logits,
        out,
        HS=hs,
        HC=hc,
        BLOCK_J=_COMBINE_BLOCK_J,
        num_warps=4,
    )
    return out
