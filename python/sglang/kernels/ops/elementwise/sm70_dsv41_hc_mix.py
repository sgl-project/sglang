"""SM70 JIT kernels for DeepSeek-V4.1 mHC mix / Sinkhorn / combine.

DSV4.1-Flash uses hc_mult=4, hidden=5120 (hc_dim=20480, mix_hc=24).
Qwen SM70 HC (`sm70_hc_mix.py`, hidden 2560 / down 320 / weight 10240) is a
different geometry and must not be called on this path.

Default path fuses mix_stats+Sinkhorn, and combine when the collapse
pre is available, into one CTA (`SGLANG_DSV41_MHC_FUSION`, default on).
hc_post stays a separate launch: it runs after attention/MLP, not at mix time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

DSV41_HC_MULT = 4
DSV41_HIDDEN = 5120
DSV41_HC_DIM = DSV41_HC_MULT * DSV41_HIDDEN  # 20480
DSV41_MIX_HC = (2 + DSV41_HC_MULT) * DSV41_HC_MULT  # 24
DSV41_SINKHORN_ITERS = 20


def dsv41_mhc_shapes_match(
    x: torch.Tensor, hc_fn: torch.Tensor, hc_mult: int = DSV41_HC_MULT
) -> bool:
    """True when x / hc_fn are the DSV4.1-Flash mHC mix geometry (fp16 act, fp32 fn)."""
    if hc_mult != DSV41_HC_MULT:
        return False
    if x.dtype != torch.float16 or hc_fn.dtype != torch.float32:
        return False
    if hc_fn.shape != (DSV41_MIX_HC, DSV41_HC_DIM):
        return False
    if x.dim() == 3:
        return x.shape[-2:] == (DSV41_HC_MULT, DSV41_HIDDEN)
    if x.dim() == 2:
        return x.shape[-1] == DSV41_HC_DIM
    return False


def _flatten_hc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        flat = x.reshape(x.shape[0], DSV41_HC_DIM)
    else:
        flat = x.reshape(x.shape[0], -1)
    return flat if flat.is_contiguous() else flat.contiguous()


def _fp32_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        t = t.float()
    return t if t.is_contiguous() else t.contiguous()


def _maybe_contig(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


@cache_once
def _module() -> Module:
    # Arch gate lives here so it runs once per process, not per forward.
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        raise RuntimeError(
            f"sm70_dsv41_hc kernels require SM70 (Volta); got SM{cap[0]}{cap[1]}"
        )
    return load_jit(
        "sm70_dsv41_hc",
        cuda_files=[
            "elementwise/sm70_dsv41_hc_mix.cuh",
            "elementwise/sm70_dsv41_hc_sinkhorn.cuh",
            "elementwise/sm70_dsv41_hc_combine.cuh",
            "elementwise/sm70_dsv41_hc_fused.cuh",
        ],
        cuda_wrappers=[
            ("mix_stats", "sm70_dsv41_hc::mix_stats"),
            ("split_sinkhorn", "sm70_dsv41_hc::split_sinkhorn"),
            ("combine", "sm70_dsv41_hc::combine"),
            ("post", "sm70_dsv41_hc::post"),
            ("mix_sinkhorn", "sm70_dsv41_hc::mix_sinkhorn"),
            ("mix_sinkhorn_combine", "sm70_dsv41_hc::mix_sinkhorn_combine"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


def _fusion_on() -> bool:
    return bool(envs.SGLANG_DSV41_MHC_FUSION.get())


def _empty_coeffs(n: int, device: torch.device):
    pre = torch.empty((n, DSV41_HC_MULT), dtype=torch.float32, device=device)
    post = torch.empty((n, DSV41_HC_MULT), dtype=torch.float32, device=device)
    comb = torch.empty(
        (n, DSV41_HC_MULT, DSV41_HC_MULT), dtype=torch.float32, device=device
    )
    return pre, post, comb


def mix_sinkhorn(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    sinkhorn_iters: int = DSV41_SINKHORN_ITERS,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """Fused mix_stats + Sinkhorn. Returns (pre, post, comb) with rank [T, ...]."""
    x_flat = _flatten_hc(x)
    n = x_flat.shape[0]
    pre, post, comb = _empty_coeffs(n, x.device)
    if n > 0:
        _module().mix_sinkhorn(
            pre,
            post,
            comb,
            x_flat,
            _maybe_contig(hc_fn),
            _fp32_contig(hc_scale),
            _fp32_contig(hc_base),
            int(sinkhorn_iters),
            float(rms_eps),
            float(hc_eps),
        )
    return pre, post, comb


def mix_sinkhorn_combine(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    combine_pre: Optional[torch.Tensor],
    *,
    sinkhorn_iters: int = DSV41_SINKHORN_ITERS,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    use_apply_pre: bool = True,
):
    """Fused mix_stats + Sinkhorn + combine. Returns (y, pre, post, comb)."""
    x_flat = _flatten_hc(x)
    n = x_flat.shape[0]
    y = torch.empty((n, DSV41_HIDDEN), dtype=x.dtype, device=x.device)
    pre, post, comb = _empty_coeffs(n, x.device)
    if use_apply_pre:
        ap = _maybe_contig(combine_pre)
        if ap.dim() > 2:
            ap = ap.reshape(n, DSV41_HC_MULT)
    else:
        ap = pre
    if n > 0:
        _module().mix_sinkhorn_combine(
            y,
            pre,
            post,
            comb,
            x_flat,
            _maybe_contig(hc_fn),
            _fp32_contig(hc_scale),
            _fp32_contig(hc_base),
            ap,
            int(sinkhorn_iters),
            float(rms_eps),
            float(hc_eps),
            1 if use_apply_pre else 0,
        )
    return y, pre, post, comb


def mix_stats(
    x: torch.Tensor, hc_fn: torch.Tensor, rms_eps: float
) -> torch.Tensor:
    """RMS-scaled mix GEMV: mixes[t] = rsqrt(mean(x^2)+eps) * (x @ hc_fn.T).

    x is fp16 [T, 4, 5120] or [T, 20480]; hc_fn is fp32 [24, 20480].
    Returns fp32 [T, 24].
    """
    x_flat = _flatten_hc(x)
    out = torch.empty(
        (x_flat.shape[0], DSV41_MIX_HC), dtype=torch.float32, device=x.device
    )
    if x_flat.shape[0] == 0:
        return out
    _module().mix_stats(out, x_flat, _maybe_contig(hc_fn), float(rms_eps))
    return out


def split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = DSV41_HC_MULT,
    sinkhorn_iters: int = DSV41_SINKHORN_ITERS,
    eps: float = 1e-6,
):
    """FP32 Sinkhorn with the same 4-wide reduction order as `_hc_split_sinkhorn_torch`.

    ``mixes`` is [B, S, 24] or [T, 24]. Returns pre/post/comb with the leading
    dims of ``mixes`` restored.
    """
    if hc_mult != DSV41_HC_MULT:
        raise RuntimeError(
            f"sm70_dsv41_hc split_sinkhorn supports hc_mult={DSV41_HC_MULT}, got {hc_mult}"
        )
    if mixes.dim() == 2:
        b, s = mixes.shape[0], 1
        flat = mixes.contiguous()
        leading = (b,)
    else:
        b, s = mixes.shape[0], mixes.shape[1]
        flat = mixes.reshape(b * s, DSV41_MIX_HC).contiguous()
        leading = (b, s)

    n = flat.shape[0]
    pre = torch.empty((n, DSV41_HC_MULT), dtype=torch.float32, device=mixes.device)
    post = torch.empty((n, DSV41_HC_MULT), dtype=torch.float32, device=mixes.device)
    comb = torch.empty(
        (n, DSV41_HC_MULT, DSV41_HC_MULT), dtype=torch.float32, device=mixes.device
    )
    if n > 0:
        _module().split_sinkhorn(
            pre,
            post,
            comb,
            flat,
            _fp32_contig(hc_scale),
            _fp32_contig(hc_base),
            int(sinkhorn_iters),
            float(eps),
        )
    if mixes.dim() == 2:
        return pre, post, comb
    return (
        pre.reshape(*leading, DSV41_HC_MULT),
        post.reshape(*leading, DSV41_HC_MULT),
        comb.reshape(*leading, DSV41_HC_MULT, DSV41_HC_MULT),
    )


def combine(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """y[t, h] = sum_k pre[t, k] * x[t, k, h]. fp16 activations, fp32 pre."""
    x_flat = _flatten_hc(x)
    y = torch.empty(
        (x_flat.shape[0], DSV41_HIDDEN), dtype=x.dtype, device=x.device
    )
    if x_flat.shape[0] == 0:
        return y
    _module().combine(y, x_flat, _maybe_contig(pre))
    return y


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """out[t,j,h] = post[t,j]*x[t,h] + sum_k comb[t,k,j]*residual[t,k,h]."""
    out = torch.empty_like(residual)
    if residual.shape[0] == 0:
        return out
    _module().post(
        out,
        _maybe_contig(x),
        _maybe_contig(residual),
        _maybe_contig(post_mix),
        _maybe_contig(comb),
    )
    return out


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    sinkhorn_iters: int = DSV41_SINKHORN_ITERS,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """SM70 hc_pre: mix GEMV + Sinkhorn + combine. Returns (y, post, comb, False)."""
    if _fusion_on():
        y, _pre, post, comb = mix_sinkhorn_combine(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            combine_pre=None,
            sinkhorn_iters=sinkhorn_iters,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            use_apply_pre=False,
        )
        return y, post, comb, False
    mixes = mix_stats(x, hc_fn, rms_eps)
    pre, post, comb = split_sinkhorn(
        mixes, hc_scale, hc_base, DSV41_HC_MULT, sinkhorn_iters, hc_eps
    )
    y = combine(x, pre)
    return y, post, comb, False


def mix_and_combine(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    apply_pre: Optional[torch.Tensor],
    norm,
    *,
    sinkhorn_iters: int = DSV41_SINKHORN_ITERS,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """SM70 `_hc_mix_and_combine`: mixes from x, collapse with apply_pre, RMSNorm.

    Returns (y, pre, post, comb) with squeezed coefficient ranks.
    """
    if _fusion_on():
        if apply_pre is None:
            pre, post, comb = mix_sinkhorn(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                sinkhorn_iters=sinkhorn_iters,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
            )
            s0 = x.select(1, 0)
            y = norm(s0 if s0.is_contiguous() else s0.contiguous())
        else:
            y, pre, post, comb = mix_sinkhorn_combine(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                apply_pre,
                sinkhorn_iters=sinkhorn_iters,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                use_apply_pre=True,
            )
            y = norm(y)
        return y, pre, post, comb
    mixes = mix_stats(x, hc_fn, rms_eps)
    pre, post, comb = split_sinkhorn(
        mixes, hc_scale, hc_base, DSV41_HC_MULT, sinkhorn_iters, hc_eps
    )
    if apply_pre is None:
        s0 = x.select(1, 0)
        y = norm(s0 if s0.is_contiguous() else s0.contiguous())
    else:
        y = norm(combine(x, apply_pre))
    return y, pre, post, comb
