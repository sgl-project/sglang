"""SM70 mHC Sinkhorn + combine / post for GLM-5.3-Flash (hidden 4096).

These kernels replace the 20-iteration Sinkhorn launch chain and the hc_post
reduction. glm_hc_pre_fused also computes the 24-wide mixes (RMS-scaled fn
GEMV) with fp64 accumulation instead of the torch fp32 chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = 4096
HC = 4
MIX = 24


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        raise RuntimeError(f"sm70_glm_hc requires SM70; got SM{cap[0]}{cap[1]}")
    return load_jit(
        "sm70_glm_hc",
        cuda_files=["elementwise/sm70_glm_hc.cuh"],
        cuda_wrappers=[
            ("pre", "sm70_glm_hc::pre"),
            ("pre_fused", "sm70_glm_hc::pre_fused"),
            ("pre_fused_norm", "sm70_glm_hc::pre_fused_norm"),
            ("post", "sm70_glm_hc::post"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


def _fp32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        t = t.float()
    return t if t.is_contiguous() else t.contiguous()


def glm_hc_pre(
    residual: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """residual [T, 4, 4096] fp16, mixes [T, 24] fp32.

    Returns post [T, 4, 1], comb [T, 4, 4], layer_input [T, 4096] fp16.
    """
    tokens = residual.shape[0]
    flat = residual.reshape(tokens, HC * HIDDEN)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    pre = torch.empty((tokens, HC), dtype=torch.float32, device=residual.device)
    post = torch.empty((tokens, HC), dtype=torch.float32, device=residual.device)
    comb = torch.empty((tokens, HC, HC), dtype=torch.float32, device=residual.device)
    layer = torch.empty((tokens, HIDDEN), dtype=torch.float16, device=residual.device)
    if tokens:
        _module().pre(
            pre,
            post,
            comb,
            layer,
            _fp32(mixes),
            flat,
            _fp32(hc_scale),
            _fp32(hc_base),
            int(sinkhorn_iters),
            float(eps),
        )
    return post.unsqueeze(-1), comb, layer


# Blocks per token for the mix GEMV; enough to spread fn's 1.5 MB over the SMs
# at decode. Measured on V100 at T=1; arbitrary for larger T.
_MIX_TARGET_BLOCKS = 64
_MIX_MAX_SPLITS = HC * HIDDEN // 256


def glm_hc_pre_fused_norm_covered(norm_weight: torch.Tensor) -> bool:
    return (
        norm_weight.dtype == torch.float16
        and norm_weight.shape == (HIDDEN,)
        and norm_weight.is_contiguous()
        and norm_weight.data_ptr() % 16 == 0
    )


def _mix_splits(tokens: int) -> int:
    # Decode and MTP verify (up to four tokens) share the one-token split, so a
    # verify row is summed in the same order as at decode.
    blocks_per_split = 1 if tokens <= 4 else tokens
    splits = 1
    while splits < _MIX_MAX_SPLITS and blocks_per_split * splits < _MIX_TARGET_BLOCKS:
        splits *= 2
    return splits


def glm_hc_pre_fused(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    sinkhorn_iters: int,
    eps: float,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """glm_hc_pre with mixes = (x @ fn.T) * rsqrt(mean(x^2) + rms_eps) computed in-kernel.

    residual [T, 4, 4096] fp16, fn [24, 16384] fp32. With norm_weight (see
    glm_hc_pre_fused_norm_covered) layer_input comes back as sgl_kernel
    rmsnorm(layer_input, norm_weight, norm_eps), bitwise.
    """
    tokens = residual.shape[0]
    flat = residual.reshape(tokens, HC * HIDDEN)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    splits = _mix_splits(tokens)
    device = residual.device
    pre = torch.empty((tokens, HC), dtype=torch.float32, device=device)
    post = torch.empty((tokens, HC), dtype=torch.float32, device=device)
    comb = torch.empty((tokens, HC, HC), dtype=torch.float32, device=device)
    layer = torch.empty((tokens, HIDDEN), dtype=torch.float16, device=device)
    partials = torch.empty((tokens, splits, MIX + 1), dtype=torch.float64, device=device)
    if tokens:
        args = (
            pre,
            post,
            comb,
            layer,
            partials,
            flat,
            _fp32(fn),
            _fp32(hc_scale),
            _fp32(hc_base),
        )
        if norm_weight is None:
            _module().pre_fused(*args, int(sinkhorn_iters), float(eps), float(rms_eps))
        else:
            _module().pre_fused_norm(
                *args,
                norm_weight,
                int(sinkhorn_iters),
                float(eps),
                float(rms_eps),
                float(norm_eps),
            )
    return post.unsqueeze(-1), comb, layer


def glm_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """x [T, 4096] fp16, residual [T, 4, 4096] fp16, post [T, 4], comb [T, 4, 4]."""
    tokens = x.shape[0]
    flat_res = residual.reshape(tokens, HC * HIDDEN)
    if not flat_res.is_contiguous():
        flat_res = flat_res.contiguous()
    out = torch.empty_like(flat_res)
    if tokens:
        _module().post(
            out,
            x if x.is_contiguous() else x.contiguous(),
            flat_res,
            _fp32(post.reshape(tokens, HC)),
            _fp32(comb.reshape(tokens, HC, HC)),
        )
    return out.view(tokens, HC, HIDDEN)
