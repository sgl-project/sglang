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
    BLOCK_K = 1024
    BLOCK_D = 1024
    hc_pow2 = triton.next_power_of_2(hc_mult)
    # One BLOCK_K tile per CTA: the partials then replay the old kernel's
    # per-tile accumulation order exactly.
    nsplit = triton.cdiv(k_total, BLOCK_K)
    part_stride = 1 + 2 * hc_pow2
    part = torch.empty((T, nsplit, part_stride), dtype=torch.float32, device=x.device)

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
