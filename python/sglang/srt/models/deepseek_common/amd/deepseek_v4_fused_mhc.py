import logging
from typing import Optional, Tuple

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip
from sglang.srt.utils.common import is_sm120_supported

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()

_FUSED_HC_POST_PRE_M_THRESHOLD = 64
_FUSED_HC_POST_PRE_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_TRITON_MHC_POST_PRE_OPS = None
_TRITON_MHC_POST_PRE_RUNTIME_DISABLED = False

# aiter fused mHC state. The kernel self-gates the fused-vs-unfused decision per
# arch internally (see aiter.ops.mhc.mhc_fused_post_pre), so this wrapper only
# tracks import/runtime availability and never re-implements the token threshold.
_AITER_MHC_FUSED_POST_PRE_RUNTIME_DISABLED = False
_AITER_MHC_IMPORT_WARNED = False


def _is_fused_mhc_post_pre_enabled() -> bool:
    # SM120 disables the standalone TileLang pre path. mhc_fused_post_pre does
    # not read that flag and dispatches independently for both small and large
    # token batches, so the standalone pre flag must not veto the fused opt-in.
    return (
        envs.SGLANG_OPT_FUSE_MHC_POST_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
        and (envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get() or is_sm120_supported())
    )


def _is_aiter_gfx95_mhc_available() -> bool:
    return _is_hip and get_bool_env_var("SGLANG_USE_AITER") and is_gfx95_supported()


def _is_production_mhc_enabled() -> bool:
    return _is_fused_mhc_post_pre_enabled() or _is_aiter_gfx95_mhc_available()


def is_cross_layer_mhc_fusion_enabled() -> bool:
    """Whether DeepSeek-V4 may defer mHC post across the attn/MoE boundary.

    Cross-layer fusion requires a fused post+pre kernel to be available: either
    the TileLang path (``SGLANG_OPT_FUSE_MHC_POST_PRE`` + TileLang pre/post) or
    the aiter HIP path on a supported gfx95 device.
    """
    return _is_production_mhc_enabled()


def _get_triton_mhc_post_pre_ops():
    global _TRITON_MHC_POST_PRE_OPS

    if _TRITON_MHC_POST_PRE_OPS is not None:
        return _TRITON_MHC_POST_PRE_OPS

    try:
        from aiter.ops.triton.fusions.mhc import mhc_post_pre
        from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config
    except Exception as err:
        logger.warning(
            "Triton fused mHC (mhc_post_pre) is unavailable, falling back: %s", err
        )
        return None

    _TRITON_MHC_POST_PRE_OPS = (mhc_post_pre, get_mhc_config)
    return _TRITON_MHC_POST_PRE_OPS


def _get_fused_hc_post_pre_buffers(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[dict[str, torch.Tensor]]:
    ops = _get_triton_mhc_post_pre_ops()
    if ops is None:
        return None
    _, get_mhc_config = ops

    key = (num_tokens, hidden_size, hc_mult, dtype, device.type, device.index)
    bufs = _FUSED_HC_POST_PRE_CACHE.get(key)
    if bufs is not None:
        return bufs

    try:
        cfg, _ = get_mhc_config("MHC_FUSED", num_tokens, hidden_size, mode="sinkhorn")
    except Exception as err:
        logger.warning("Failed to initialize fused mHC config, falling back: %s", err)
        return None

    n_total = 2 * hc_mult + hc_mult * hc_mult
    k_dim = hc_mult * hidden_size
    block_k = cfg.get("BLOCK_K", min(512, triton.next_power_of_2(k_dim)))
    block_k = min(block_k, triton.next_power_of_2(k_dim))
    block_c_split = max(block_k // hc_mult, 1)
    num_ksplit = triton.cdiv(hidden_size, block_c_split)

    bufs = {
        "residual_out": torch.empty(
            num_tokens, hc_mult, hidden_size, dtype=dtype, device=device
        ),
        "layer_input_out": torch.empty(
            num_tokens, hidden_size, dtype=dtype, device=device
        ),
        "h_post": torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device),
        "h_res": torch.empty(
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device
        ),
        "acc_partial": torch.empty(
            num_ksplit, num_tokens, n_total, dtype=torch.float32, device=device
        ),
        "acc_sq_partial": torch.empty(
            num_ksplit, num_tokens, dtype=torch.float32, device=device
        ),
    }
    _FUSED_HC_POST_PRE_CACHE[key] = bufs
    return bufs


def try_fused_hc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn_t: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    is_gfx95_supported: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    global _TRITON_MHC_POST_PRE_RUNTIME_DISABLED

    if (
        _TRITON_MHC_POST_PRE_RUNTIME_DISABLED
        or not is_gfx95_supported
        or x.shape[0] == 0
        or x.shape[0] > _FUSED_HC_POST_PRE_M_THRESHOLD
        or x.dim() != 2
        or residual.dim() != 3
    ):
        return None

    ops = _get_triton_mhc_post_pre_ops()
    if ops is None:
        return None
    mhc_post_pre, _ = ops

    bufs = _get_fused_hc_post_pre_buffers(
        x.shape[0], x.shape[1], hc_mult, residual.dtype, x.device
    )
    if bufs is None:
        return None

    try:
        _, _, layer_input_out, new_residual = mhc_post_pre(
            x,
            residual,
            post,
            comb,
            hc_fn_t,
            hc_scale,
            hc_base,
            hc_mult,
            norm_eps,
            hc_eps,
            hc_post_mult,
            sinkhorn_iters,
            # Match sglang's exp-domain asymmetric Sinkhorn used in hc_pre.
            asymmetric_exp_domain=True,
            hc_sinkhorn_eps=hc_eps,
            residual_out=bufs["residual_out"],
            h_post=bufs["h_post"],
            h_res=bufs["h_res"],
            layer_input_out=bufs["layer_input_out"],
            acc_partial=bufs["acc_partial"],
            acc_sq_partial=bufs["acc_sq_partial"],
        )
    except Exception as err:
        logger.warning(
            "Triton fused mHC kernel failed, disabling fallback path: %s", err
        )
        _TRITON_MHC_POST_PRE_RUNTIME_DISABLED = True
        return None

    return new_residual, layer_input_out, bufs["h_post"], bufs["h_res"], False


def try_aiter_fused_mhc_post_pre(
    layer_input: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    norm_weight: Optional[torch.Tensor],
    norm_eps: Optional[float],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    """Fused mhc_post + next-layer mhc_pre via the aiter HIP kernel.

    Returns ``(next_residual, layer_input, post_mix, comb_mix, norm_applied)`` or
    ``None`` to let the caller fall back. The aiter kernel internally chooses
    between its fused and unfused (mhc_post + mhc_pre) implementations based on the
    token count and detected arch, so no token threshold is applied here.
    """
    global _AITER_MHC_FUSED_POST_PRE_RUNTIME_DISABLED, _AITER_MHC_IMPORT_WARNED

    if (
        _AITER_MHC_FUSED_POST_PRE_RUNTIME_DISABLED
        or not _is_aiter_gfx95_mhc_available()
        or layer_input.shape[0] == 0
        or layer_input.dim() != 2
        or residual.dim() != 3
    ):
        return None

    try:
        from aiter.ops.mhc import mhc_fused_post_pre
    except Exception as err:
        if not _AITER_MHC_IMPORT_WARNED:
            logger.warning("aiter fused mHC is unavailable, falling back: %s", err)
            _AITER_MHC_IMPORT_WARNED = True
        _AITER_MHC_FUSED_POST_PRE_RUNTIME_DISABLED = True
        return None

    norm_kwargs = {}
    if norm_weight is not None:
        norm_kwargs["norm_weight"] = norm_weight
        norm_kwargs["norm_eps"] = norm_eps if norm_eps is not None else rms_eps

    try:
        post_mix, comb_mix, layer_input_out, next_residual = mhc_fused_post_pre(
            layer_input=layer_input,
            residual_in=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_eps,
            hc_sinkhorn_eps=hc_eps,
            hc_post_mult_value=hc_post_mult,
            sinkhorn_repeat=sinkhorn_iters,
            **norm_kwargs,
        )
    except Exception as err:
        logger.warning(
            "aiter fused mHC kernel failed, disabling fallback path: %s", err
        )
        _AITER_MHC_FUSED_POST_PRE_RUNTIME_DISABLED = True
        return None

    post_out = post_mix.squeeze(-1) if post_mix.ndim == 3 else post_mix
    return (
        next_residual,
        layer_input_out,
        post_out,
        comb_mix,
        norm_weight is not None,
    )


def try_mhc_fused_post_pre_boundary(
    layer_input: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    norm_weight: Optional[torch.Tensor],
    norm_eps: Optional[float],
    fn_transpose: bool,
    is_gfx95_supported_flag: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    """Dispatch the fused mHC post+pre across the attn/MoE boundary.

    Preference order (first available wins): aiter HIP kernel, then the Triton
    kernel. Returns ``None`` when neither fires so the caller can fall back to the
    TileLang path or the unfused ``hc_post`` + ``hc_pre`` sequence.

    The aiter and Triton kernels expect opposite ``fn`` orientations: the Triton
    kernel takes ``hc_fn`` transposed (``fn_transpose``), while aiter consumes it
    in the native ``mhc_pre`` layout.
    """
    aiter_result = try_aiter_fused_mhc_post_pre(
        layer_input,
        residual,
        post,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_post_mult,
        sinkhorn_iters,
        norm_weight,
        norm_eps,
    )
    if aiter_result is not None:
        return aiter_result

    triton_fn = hc_fn.T if fn_transpose else hc_fn
    return try_fused_hc_post_pre(
        layer_input,
        residual,
        post,
        comb,
        triton_fn,
        hc_scale,
        hc_base,
        hc_mult,
        rms_eps,
        hc_eps,
        hc_post_mult,
        sinkhorn_iters,
        is_gfx95_supported_flag,
    )


def apply_mhc_post_pre_boundary(
    layer_input: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    norm_weight: Optional[torch.Tensor],
    norm_eps: Optional[float],
    *,
    fn_transpose: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    # Try the aiter/Triton fused post+pre kernels first; if neither fires,
    # fall back to the TileLang fused kernel, else return None so the caller
    # runs the unfused hc_post + hc_pre sequence.
    fused = try_mhc_fused_post_pre_boundary(
        layer_input,
        residual,
        post,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        rms_eps,
        hc_eps,
        hc_post_mult,
        sinkhorn_iters,
        norm_weight,
        norm_eps,
        fn_transpose,
        _is_gfx95_supported,
    )
    if fused is not None:
        return fused

    if not _is_fused_mhc_post_pre_enabled():
        return None

    from sglang.srt.models.deepseek_v4 import _get_mhc_ops

    post_in = post.unsqueeze(-1) if post.ndim == 2 else post
    (
        residual,
        post_out,
        comb_out,
        layer_input_out,
    ) = _get_mhc_ops().mhc_fused_post_pre(
        layer_input,
        residual,
        post_in,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        hc_post_mult,
        sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    post_out = post_out.squeeze(-1) if post_out.ndim == 3 else post_out
    return residual, layer_input_out, post_out, comb_out, True
