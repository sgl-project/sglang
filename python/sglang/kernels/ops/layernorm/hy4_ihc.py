from __future__ import annotations

import functools
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_HPC_IHC_CAPABILITIES = ((9, 0), (10, 0), (10, 3))
_HPC_IHC_HC_MULTS = (4,)
_HPC_IHC_HIDDEN_SIZES = (4096, 6144)


@functools.lru_cache(maxsize=None)
def _hpc_ihc_op(op_name: str, hc_mult: int, hidden_size: int):
    try:
        import hpc
    except ImportError:
        return None

    op = getattr(hpc, op_name, None)
    if op is None:
        logger.info(
            "HY4 iHC: the installed hpc build (%s) has no %s; using the "
            "in-tree Triton kernels.",
            getattr(hpc, "__version__", "unknown"),
            op_name,
        )
        return None

    from sglang.srt.utils import get_device_capability

    cap = get_device_capability()
    if cap not in _HPC_IHC_CAPABILITIES:
        logger.warning(
            "HY4 iHC: hpc.%s is unavailable on sm%s%s.",
            op_name,
            *cap,
        )
        return None
    if hc_mult not in _HPC_IHC_HC_MULTS or hidden_size not in _HPC_IHC_HIDDEN_SIZES:
        logger.warning(
            "HY4 iHC: hpc.%s is instantiated for hc_mult in %s and hidden_size "
            "in %s, got %d / %d.",
            op_name,
            _HPC_IHC_HC_MULTS,
            _HPC_IHC_HIDDEN_SIZES,
            hc_mult,
            hidden_size,
        )
        return None

    logger.info("HY4 iHC: using hpc.%s.", op_name)
    return op


@triton.jit
def _hy4_ihc_pre_stage1(
    x_ptr,
    fn_ptr,
    part_ptr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PART_STRIDE: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_s = tl.program_id(1)
    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT

    k_offs = pid_s * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K_TOTAL

    x_tile = tl.load(x_ptr + pid_t * K_TOTAL + k_offs, mask=k_mask, other=0.0).to(
        tl.float32
    )
    sumsq = tl.sum(x_tile * x_tile, axis=0)

    fn_offs = m_idx[:, None] * K_TOTAL + k_offs[None, :]
    fn_mask = m_mask[:, None] & k_mask[None, :]
    mix_pre = tl.sum(
        tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0) * x_tile[None, :], axis=1
    )
    mix_post = tl.sum(
        tl.load(fn_ptr + HC_MULT * K_TOTAL + fn_offs, mask=fn_mask, other=0.0)
        * x_tile[None, :],
        axis=1,
    )

    base = part_ptr + (pid_t * NSPLIT + pid_s) * PART_STRIDE
    tl.store(base, sumsq)
    tl.store(base + 1 + m_idx, mix_pre, mask=m_mask)
    tl.store(base + 1 + HC_POW2 + m_idx, mix_post, mask=m_mask)


@triton.jit
def _ihc_dot_partials(
    x_bf16,
    fn_ptr,
    k_offs,
    k_mask,
    base,
    t_mask,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """The 2 * HC_MULT hc_fn dot products of a [BLOCK_T, BLOCK_K] bf16 tile on
    tensor cores: the fp32 hc_fn tile is split into three bf16 terms whose sum
    is hc_fn to ~fp32 precision, so each product is exact in fp32."""
    n = tl.arange(0, 16)
    rest = tl.load(
        fn_ptr + n[None, :] * K_TOTAL + k_offs[:, None],
        mask=k_mask[:, None] & (n < 2 * HC_MULT)[None, :],
        other=0.0,
    )
    acc = tl.zeros((BLOCK_T, 16), dtype=tl.float32)
    for i in tl.static_range(3):
        w = rest.to(tl.bfloat16)
        rest = rest - w.to(tl.float32)
        acc = tl.dot(x_bf16, w, acc)
    col = tl.where(n < HC_MULT, 1 + n, 1 + HC_POW2 + n - HC_MULT)
    tl.store(
        base[:, None] + col[None, :],
        acc,
        mask=t_mask[:, None] & (n < 2 * HC_MULT)[None, :],
    )


@triton.jit
def _hy4_ihc_pre_stage1_tokens(
    x_ptr,
    fn_ptr,
    part_ptr,
    T,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    USE_DOT: tl.constexpr = False,
):
    """Stage 1 for many tokens: BLOCK_T tokens share each hc_fn tile instead of
    every token re-reading it. Writes the same per-(token, tile) partials."""
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    t_offs = (pid_t * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    t_mask = t_offs < T
    k_offs = pid_s * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K_TOTAL
    x_raw = tl.load(
        x_ptr + t_offs[:, None] * K_TOTAL + k_offs[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    )
    x_tile = x_raw.to(tl.float32)
    base = part_ptr + (t_offs * NSPLIT + pid_s) * PART_STRIDE
    tl.store(base, tl.sum(x_tile * x_tile, axis=1), mask=t_mask)
    if USE_DOT:
        _ihc_dot_partials(
            x_raw,
            fn_ptr,
            k_offs,
            k_mask,
            base,
            t_mask,
            K_TOTAL,
            HC_MULT,
            HC_POW2,
            BLOCK_T,
        )
        return
    for m in tl.static_range(HC_MULT):
        w_pre = tl.load(fn_ptr + m * K_TOTAL + k_offs, mask=k_mask, other=0.0)
        w_post = tl.load(
            fn_ptr + (HC_MULT + m) * K_TOTAL + k_offs, mask=k_mask, other=0.0
        )
        tl.store(base + 1 + m, tl.sum(x_tile * w_pre[None, :], axis=1), mask=t_mask)
        tl.store(
            base + 1 + HC_POW2 + m,
            tl.sum(x_tile * w_post[None, :], axis=1),
            mask=t_mask,
        )


@triton.jit
def _hy4_ihc_pre_stage2(
    x_ptr,
    part_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    hidden_size: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    magnitude: tl.constexpr,
    norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT

    # The single-CTA kernel folded the BLOCK_K tiles into one accumulator with
    # '+='; replay that ascending order over the partials so the fp32 sum is
    # bit-identical.
    row = part_ptr + pid_t * NSPLIT * PART_STRIDE
    sumsq = tl.zeros((), dtype=tl.float32)
    mix_pre = tl.zeros((HC_POW2,), dtype=tl.float32)
    mix_post = tl.zeros((HC_POW2,), dtype=tl.float32)
    for s in tl.static_range(NSPLIT):
        b = row + s * PART_STRIDE
        sumsq += tl.load(b)
        mix_pre += tl.load(b + 1 + m_idx, mask=m_mask, other=0.0)
        mix_post += tl.load(b + 1 + HC_POW2 + m_idx, mask=m_mask, other=0.0)

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + norm_eps)
    scale_pre = tl.load(scale_ptr)
    scale_post = tl.load(scale_ptr + 1)
    base_pre = tl.load(base_ptr + m_idx, mask=m_mask, other=0.0)
    base_post = tl.load(base_ptr + HC_MULT + m_idx, mask=m_mask, other=0.0)

    pre = tl.sigmoid(mix_pre * rsqrt * scale_pre + base_pre) + hc_eps
    if pid_d == 0:
        post = (
            magnitude * tl.sigmoid(mix_post * rsqrt * scale_post + base_post) + hc_eps
        )
        tl.store(post_ptr + pid_t * HC_MULT + m_idx, post, mask=m_mask)

    x_row = x_ptr + pid_t * K_TOTAL
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_size
    y_block = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for m in tl.static_range(HC_MULT):
        x_m = tl.load(x_row + m * hidden_size + d_offs, mask=d_mask, other=0.0)
        pre_m = tl.sum(tl.where(m_idx == m, pre, 0.0), axis=0)
        y_block += pre_m * x_m.to(tl.float32)
    tl.store(
        y_ptr + pid_t * hidden_size + d_offs,
        y_block.to(y_ptr.dtype.element_ty),
        mask=d_mask,
    )


@triton.jit
def _hy4_ihc_post_kernel(
    out_ptr,
    res_ptr,
    post_ptr,
    y_ptr,
    hidden_size: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)

    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT
    post = tl.load(post_ptr + pid_t * HC_MULT + m_idx, mask=m_mask, other=0.0)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_size
    out_block = tl.load(out_ptr + pid_t * hidden_size + d_offs, mask=d_mask, other=0.0)
    out_block = out_block.to(tl.float32)

    res_row = res_ptr + pid_t * HC_MULT * hidden_size
    y_row = y_ptr + pid_t * HC_MULT * hidden_size
    for m in tl.static_range(HC_MULT):
        res_block = tl.load(res_row + m * hidden_size + d_offs, mask=d_mask, other=0.0)
        post_m = tl.sum(tl.where(m_idx == m, post, 0.0), axis=0)
        y_block = post_m * out_block + res_block.to(tl.float32)
        tl.store(
            y_row + m * hidden_size + d_offs,
            y_block.to(y_ptr.dtype.element_ty),
            mask=d_mask,
        )


# Below this many tokens the per-token stage 1 (one CTA per token tile) wins.
_IHC_TOKEN_BLOCKED_MIN_T = 64
_IHC_TOKEN_BLOCK_T = 32
_IHC_TOKEN_BLOCK_K = 256
_IHC_TOKEN_WARPS = 8


def fused_hy4_ihc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3, f"x must be 3D (T, hc_mult, hidden_size), got {x.shape}"
    assert hc_fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32

    x = x.contiguous()
    hc_fn = hc_fn.contiguous()
    T, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    assert hc_fn.shape == (2 * hc_mult, k_total)
    assert hc_base.shape == (2 * hc_mult,)
    assert hc_scale.shape == (2,)

    if T == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
        )

    hpc_op = _hpc_ihc_op("fuse_ihc_pre", hc_mult, hidden_size)
    if hpc_op is not None:
        return hpc_op(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps,
            hc_eps,
            magnitude,
            rms_weight,
            rms_eps,
            True,
        )

    y = torch.empty((T, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((T, hc_mult), dtype=torch.float32, device=x.device)
    token_blocked = T >= _IHC_TOKEN_BLOCKED_MIN_T
    BLOCK_K = _IHC_TOKEN_BLOCK_K if token_blocked else 1024
    BLOCK_D = 1024
    hc_pow2 = triton.next_power_of_2(hc_mult)
    # One BLOCK_K tile per CTA: the partials then replay the old kernel's
    # per-tile accumulation order exactly.
    nsplit = triton.cdiv(k_total, BLOCK_K)
    part_stride = 1 + 2 * hc_pow2
    part = torch.empty((T, nsplit, part_stride), dtype=torch.float32, device=x.device)

    if token_blocked:
        # Prefill-sized batches: reuse each hc_fn tile across a block of tokens.
        block_t = _IHC_TOKEN_BLOCK_T
        _hy4_ihc_pre_stage1_tokens[(triton.cdiv(T, block_t), nsplit)](
            x,
            hc_fn,
            part,
            T,
            K_TOTAL=k_total,
            HC_MULT=hc_mult,
            HC_POW2=hc_pow2,
            NSPLIT=nsplit,
            BLOCK_K=BLOCK_K,
            BLOCK_T=block_t,
            PART_STRIDE=part_stride,
            num_warps=_IHC_TOKEN_WARPS,
            enable_fp_fusion=False,
        )
    else:
        _hy4_ihc_pre_stage1[(T, nsplit)](
            x,
            hc_fn,
            part,
            K_TOTAL=k_total,
            HC_MULT=hc_mult,
            HC_POW2=hc_pow2,
            NSPLIT=nsplit,
            BLOCK_K=BLOCK_K,
            PART_STRIDE=part_stride,
            num_warps=8,
            # Disable FMA: eager rounds the fp32 product before summation.
            enable_fp_fusion=False,
        )
    _hy4_ihc_pre_stage2[(T, triton.cdiv(hidden_size, BLOCK_D))](
        x,
        part,
        hc_scale,
        hc_base,
        y,
        post,
        hidden_size=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NSPLIT=nsplit,
        PART_STRIDE=part_stride,
        BLOCK_D=BLOCK_D,
        magnitude=magnitude,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if rms_weight is not None:
        y_float = y.float()
        y = (
            y_float
            * torch.rsqrt(y_float.square().mean(dim=-1, keepdim=True) + rms_eps)
            * rms_weight.float()
        ).to(y.dtype)
    return y, post


def fused_hy4_ihc_post(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    assert output.dim() == 2, f"output must be 2D (T, hidden_size), got {output.shape}"
    assert post.dtype == torch.float32

    output = output.contiguous()
    residual = residual.contiguous()
    T, hidden_size = output.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (T, hc_mult, hidden_size)
    assert post.shape == (T, hc_mult)

    if T == 0:
        return torch.empty(
            (0, hc_mult, hidden_size), dtype=output.dtype, device=output.device
        )

    hpc_op = _hpc_ihc_op("fuse_ihc_post", hc_mult, hidden_size)
    if hpc_op is not None:
        return hpc_op(output, residual, post)

    y = torch.empty((T, hc_mult, hidden_size), dtype=output.dtype, device=output.device)
    BLOCK_D = 1024
    grid = (T, triton.cdiv(hidden_size, BLOCK_D))
    _hy4_ihc_post_kernel[grid](
        output,
        residual,
        post.contiguous(),
        y,
        hidden_size=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        BLOCK_D=BLOCK_D,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


def fused_hy4_ihc_post_pre(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hidden_size = output.shape
    hc_mult = residual.shape[1]
    if num_tokens == 0:
        return (
            torch.empty_like(residual),
            torch.empty_like(output),
            torch.empty((0, hc_mult), dtype=torch.float32, device=output.device),
        )

    hpc_op = _hpc_ihc_op("fuse_ihc_post_pre", hc_mult, hidden_size)
    if hpc_op is not None:
        return hpc_op(
            output.contiguous(),
            residual.contiguous(),
            post.contiguous(),
            hc_fn.contiguous(),
            hc_scale.contiguous(),
            hc_base.contiguous(),
            norm_eps,
            hc_eps,
            magnitude,
            rms_weight,
            rms_eps,
            True,
        )

    next_residual = fused_hy4_ihc_post(output, residual, post)
    reduced, next_post = fused_hy4_ihc_pre(
        next_residual,
        hc_fn,
        hc_scale,
        hc_base,
        magnitude,
        norm_eps,
        hc_eps,
        rms_weight,
        rms_eps,
    )
    return next_residual, reduced, next_post


def fused_hy4_ihc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 0.0,
) -> torch.Tensor:
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return torch.empty(
            (0, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
        )

    hpc_op = _hpc_ihc_op("fuse_ihc_head", hc_mult, hidden_size)
    if hpc_op is not None:
        return hpc_op(
            hidden_states.contiguous(),
            hc_fn.contiguous(),
            hc_scale.contiguous(),
            hc_base.contiguous(),
            norm_eps,
            hc_eps,
            rms_weight,
            rms_eps,
            True,
        )

    flat = hidden_states.flatten(1).float()
    scale = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    gates = torch.nn.functional.linear(flat, hc_fn) * scale
    gates = torch.sigmoid(gates * hc_scale + hc_base) + hc_eps
    output = torch.sum(gates.unsqueeze(-1) * hidden_states.float(), dim=1).to(
        hidden_states.dtype
    )
    if rms_weight is not None:
        output_float = output.float()
        output = (
            output_float
            * torch.rsqrt(output_float.square().mean(dim=-1, keepdim=True) + rms_eps)
            * rms_weight.float()
        ).to(output.dtype)
    return output


@triton.jit
def _hy4_ihc_post_stage1(
    out_ptr,
    res_ptr,
    post_ptr,
    new_res_ptr,
    fn_ptr,
    part_ptr,
    out_w_ptr,
    T,
    out_eps,
    HIDDEN: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    OUT_NORM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_DOT: tl.constexpr = False,
):
    """Stage 1 with the previous block's iHC post folded in: builds the new
    residual tile `post * out + residual`, stores it, and reduces it. With
    OUT_NORM, `out` first goes through the block's output RMSNorm (each CTA
    re-reads the whole `out` row, so this is for small token counts)."""
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    t_offs = (pid_t * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    t_mask = t_offs < T
    k_offs = pid_s * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K_TOTAL
    mask = t_mask[:, None] & k_mask[None, :]
    if HIDDEN % BLOCK_K == 0:
        # The tile lies in one stream: one post gate per token.
        stream = (pid_s * BLOCK_K) // HIDDEN
        d_offs = pid_s * BLOCK_K - stream * HIDDEN + tl.arange(0, BLOCK_K)
        gate = tl.load(post_ptr + t_offs * HC_MULT + stream, mask=t_mask, other=0.0)[
            :, None
        ]
    else:
        stream = k_offs // HIDDEN
        d_offs = k_offs % HIDDEN
        gate = tl.load(
            post_ptr + t_offs[:, None] * HC_MULT + stream[None, :], mask=mask, other=0.0
        )
    out = tl.load(
        out_ptr + t_offs[:, None] * HIDDEN + d_offs[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if OUT_NORM:
        h_offs = tl.arange(0, BLOCK_H)
        row = tl.load(
            out_ptr + t_offs[:, None] * HIDDEN + h_offs[None, :],
            mask=t_mask[:, None] & (h_offs < HIDDEN)[None, :],
            other=0.0,
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(row * row, axis=1) / HIDDEN + out_eps)
        w = tl.load(out_w_ptr + d_offs, mask=k_mask, other=0.0).to(tl.float32)
        # The unfused norm writes bf16 before the post reads it.
        out = (out * inv[:, None] * w[None, :]).to(new_res_ptr.dtype.element_ty)
        out = out.to(tl.float32)
    res = tl.load(
        res_ptr + t_offs[:, None] * K_TOTAL + k_offs[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    new_res = (gate * out + res).to(new_res_ptr.dtype.element_ty)
    tl.store(
        new_res_ptr + t_offs[:, None] * K_TOTAL + k_offs[None, :], new_res, mask=mask
    )

    x_tile = new_res.to(tl.float32)
    base = part_ptr + (t_offs * NSPLIT + pid_s) * PART_STRIDE
    tl.store(base, tl.sum(x_tile * x_tile, axis=1), mask=t_mask)
    if USE_DOT:
        _ihc_dot_partials(
            new_res,
            fn_ptr,
            k_offs,
            k_mask,
            base,
            t_mask,
            K_TOTAL,
            HC_MULT,
            HC_POW2,
            BLOCK_T,
        )
        return
    for m in tl.static_range(HC_MULT):
        w_pre = tl.load(fn_ptr + m * K_TOTAL + k_offs, mask=k_mask, other=0.0)
        w_post = tl.load(
            fn_ptr + (HC_MULT + m) * K_TOTAL + k_offs, mask=k_mask, other=0.0
        )
        tl.store(base + 1 + m, tl.sum(x_tile * w_pre[None, :], axis=1), mask=t_mask)
        tl.store(
            base + 1 + HC_POW2 + m,
            tl.sum(x_tile * w_post[None, :], axis=1),
            mask=t_mask,
        )


@triton.jit
def _hy4_ihc_pre_stage2_norm(
    x_ptr,
    part_ptr,
    scale_ptr,
    base_ptr,
    w_ptr,
    y_ptr,
    post_ptr,
    hidden_size: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NSPLIT: tl.constexpr,
    PART_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    magnitude: tl.constexpr,
    norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    rms_eps: tl.constexpr,
):
    """Stage 2 over the whole hidden row, then the following RMSNorm."""
    pid_t = tl.program_id(0).to(tl.int64)
    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT

    row = part_ptr + pid_t * NSPLIT * PART_STRIDE
    sumsq = tl.zeros((), dtype=tl.float32)
    mix_pre = tl.zeros((HC_POW2,), dtype=tl.float32)
    mix_post = tl.zeros((HC_POW2,), dtype=tl.float32)
    for s in tl.static_range(NSPLIT):
        b = row + s * PART_STRIDE
        sumsq += tl.load(b)
        mix_pre += tl.load(b + 1 + m_idx, mask=m_mask, other=0.0)
        mix_post += tl.load(b + 1 + HC_POW2 + m_idx, mask=m_mask, other=0.0)

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + norm_eps)
    scale_pre = tl.load(scale_ptr)
    scale_post = tl.load(scale_ptr + 1)
    base_pre = tl.load(base_ptr + m_idx, mask=m_mask, other=0.0)
    base_post = tl.load(base_ptr + HC_MULT + m_idx, mask=m_mask, other=0.0)
    pre = tl.sigmoid(mix_pre * rsqrt * scale_pre + base_pre) + hc_eps
    post = magnitude * tl.sigmoid(mix_post * rsqrt * scale_post + base_post) + hc_eps
    tl.store(post_ptr + pid_t * HC_MULT + m_idx, post, mask=m_mask)

    x_row = x_ptr + pid_t * K_TOTAL
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_size
    y = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for m in tl.static_range(HC_MULT):
        x_m = tl.load(x_row + m * hidden_size + d_offs, mask=d_mask, other=0.0)
        pre_m = tl.sum(tl.where(m_idx == m, pre, 0.0), axis=0)
        y += pre_m * x_m.to(tl.float32)
    # The unfused path rounds the iHC output before the norm reads it.
    y = y.to(y_ptr.dtype.element_ty).to(tl.float32)
    inv = tl.rsqrt(tl.sum(y * y, axis=0) / hidden_size + rms_eps)
    w = tl.load(w_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(
        y_ptr + pid_t * hidden_size + d_offs,
        (y * inv * w).to(y_ptr.dtype.element_ty),
        mask=d_mask,
    )


# From this many tokens, stage 1 runs its hc_fn dot products on tensor cores
# (B300, 8192 tokens: 153 -> 72 us); decode-sized batches keep the reductions.
_IHC_DOT_MIN_T = 128
_IHC_DOT_BLOCK_T = 128
_IHC_DOT_BLOCK_K = 128
# The post-fused stage 1 also holds the residual tile: smaller tiles.
_IHC_DOT_POST_CONFIG = (64, 256, 8)  # (BLOCK_T, BLOCK_K, num_warps)


def ihc_fuses_out_norm(num_tokens: int) -> bool:
    """Whether `hy4_ihc_post_pre_norm` can take the output RMSNorm for this many
    tokens."""
    return num_tokens < _IHC_TOKEN_BLOCKED_MIN_T


def hy4_ihc_post_pre_norm(
    output: torch.Tensor | None,
    residual: torch.Tensor,
    post: torch.Tensor | None,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor,
    rms_eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """iHC post (when `output` is given) + iHC pre + RMSNorm in two launches.

    Returns (new residual, normalized block input, new post gates). Without
    `output` the residual is used as is and returned unchanged.
    `out_norm_weight` applies the previous block's output RMSNorm to `output`
    first; only for token counts where `ihc_fuses_out_norm` holds.
    """
    residual = residual.contiguous()
    hc_fn = hc_fn.contiguous()
    T, hc_mult, hidden_size = residual.shape
    k_total = hc_mult * hidden_size
    assert hc_fn.dtype == torch.float32 and hc_fn.shape == (2 * hc_mult, k_total)
    y = torch.empty((T, hidden_size), dtype=residual.dtype, device=residual.device)
    next_post = torch.empty((T, hc_mult), dtype=torch.float32, device=residual.device)
    if T == 0:
        return residual, y, next_post

    token_blocked = T >= _IHC_TOKEN_BLOCKED_MIN_T
    if out_norm_weight is not None and (output is None or token_blocked):
        raise ValueError("the output RMSNorm is fused only into a small-batch post")
    use_dot = (
        T >= _IHC_DOT_MIN_T and 2 * hc_mult <= 16 and residual.dtype == torch.bfloat16
    )
    if use_dot:
        block_k = _IHC_DOT_POST_CONFIG[1] if output is not None else _IHC_DOT_BLOCK_K
    else:
        block_k = _IHC_TOKEN_BLOCK_K if token_blocked else 1024
    hc_pow2 = triton.next_power_of_2(hc_mult)
    nsplit = triton.cdiv(k_total, block_k)
    part_stride = 1 + 2 * hc_pow2
    part = torch.empty(
        (T, nsplit, part_stride), dtype=torch.float32, device=residual.device
    )
    common = dict(
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NSPLIT=nsplit,
        BLOCK_K=block_k,
        PART_STRIDE=part_stride,
        enable_fp_fusion=False,
    )
    if output is not None:
        new_residual = torch.empty_like(residual)
        block_t = (
            _IHC_DOT_POST_CONFIG[0]
            if use_dot
            else (_IHC_TOKEN_BLOCK_T if token_blocked else 1)
        )
        warps = (
            _IHC_DOT_POST_CONFIG[2]
            if use_dot
            else (_IHC_TOKEN_WARPS if token_blocked else 8)
        )
        _hy4_ihc_post_stage1[(triton.cdiv(T, block_t), nsplit)](
            output.contiguous(),
            residual,
            post.contiguous(),
            new_residual,
            hc_fn,
            part,
            out_norm_weight if out_norm_weight is not None else hc_fn,
            T,
            out_norm_eps,
            HIDDEN=hidden_size,
            BLOCK_T=block_t,
            OUT_NORM=out_norm_weight is not None,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            USE_DOT=use_dot,
            num_warps=warps,
            **common,
        )
    else:
        new_residual = residual
        if token_blocked:
            block_t = _IHC_DOT_BLOCK_T if use_dot else _IHC_TOKEN_BLOCK_T
            _hy4_ihc_pre_stage1_tokens[(triton.cdiv(T, block_t), nsplit)](
                residual,
                hc_fn,
                part,
                T,
                BLOCK_T=block_t,
                num_warps=_IHC_TOKEN_WARPS,
                USE_DOT=use_dot,
                **common,
            )
        else:
            _hy4_ihc_pre_stage1[(T, nsplit)](
                residual, hc_fn, part, num_warps=8, **common
            )
    _hy4_ihc_pre_stage2_norm[(T,)](
        new_residual,
        part,
        hc_scale,
        hc_base,
        rms_weight,
        y,
        next_post,
        hidden_size=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NSPLIT=nsplit,
        PART_STRIDE=part_stride,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        magnitude=magnitude,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        rms_eps=rms_eps,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return new_residual, y, next_post
