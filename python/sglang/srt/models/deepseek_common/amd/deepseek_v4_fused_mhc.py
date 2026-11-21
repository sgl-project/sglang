import logging
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch
import triton

if TYPE_CHECKING:
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import HcCoefficients

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.utils import should_skip_post_experts_all_reduce
from sglang.srt.runtime_context import (
    get_exec,
    get_forward,
    get_parallel,
    get_platform,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_gfx942_supported,
    is_hip,
)
from sglang.srt.utils.device import is_gfx1250_supported

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_gfx942_supported = is_gfx942_supported()
_is_gfx1250_supported = is_gfx1250_supported()

if _is_hip:
    from sglang.kernels.ops.communication.all_reduce_mhc_hip import (
        all_reduce_mhc_post,
    )
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        hc_boundary_fused_deferred,
    )
if _is_gfx95_supported:
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        Fp8GridActivation,
        Mxfp8Activation,
    )
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_gfx95_dense import (
        _ffn_norm_emits_mxfp8,
        _mxfp8_consumer,
        post_attention_norm,
    )


class MhcPostOperands(NamedTuple):
    """What the fused all-reduce + hc_post kernel reads besides the sublayer rows.

    The caller hands this to `apply_attention_mhc` / `apply_moe_mhc` directly; the
    sublayer only has to be told to leave its reduce alone.
    """

    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


# the fused all-reduce + hc_post kernel: DeepSeek-V4.1's hidden size, up to this many rows
MHC_HIDDEN_SIZE = 5120
ALL_REDUCE_MHC_MAX_ROWS = 8

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
    # gfx1250: TileLang doesn't compile; the fused cross-layer path routes
    # entirely through the Triton mhc_post_pre (try_fused_hc_post_pre).
    # Gate only on SGLANG_OPT_FUSE_MHC_POST_PRE; TileLang switches don't apply.
    if _is_gfx1250_supported:
        return envs.SGLANG_OPT_FUSE_MHC_POST_PRE.get()
    # gfx942: the TileLang fused kernel (tl.get_lane_idx) does not build;
    # cross-layer fusion routes through the aiter HIP mhc_fused_post_pre.
    if _is_gfx942_supported:
        return False
    # SM120 disables the standalone TileLang pre path. mhc_fused_post_pre does
    # not read that flag and dispatches independently for both small and large
    # token batches, so the standalone pre flag must not veto the fused opt-in.
    return (
        envs.SGLANG_OPT_FUSE_MHC_POST_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
        and (envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get() or get_platform().is_sm120)
    )


def _is_aiter_gfx95_mhc_available() -> bool:
    return (
        _is_hip
        and get_bool_env_var("SGLANG_USE_AITER")
        and (is_gfx95_supported() or is_gfx942_supported())
    )


def _is_production_mhc_enabled() -> bool:
    return _is_fused_mhc_post_pre_enabled() or _is_aiter_gfx95_mhc_available()


def is_cross_layer_mhc_fusion_enabled() -> bool:
    """Whether DeepSeek-V4 may defer mHC post across the attn/MoE boundary.

    Cross-layer fusion requires a fused post+pre kernel to be available: either
    the TileLang path (``SGLANG_OPT_FUSE_MHC_POST_PRE`` + TileLang pre/post) or
    the aiter HIP path on gfx95 or gfx942.
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
    # Cache only small batches: a caller that runs this path for every prefill
    # size (gfx1250) would otherwise keep one buffer set per num_tokens forever.
    if num_tokens <= _FUSED_HC_POST_PRE_M_THRESHOLD:
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
        or not (is_gfx95_supported or _is_gfx1250_supported)
        or x.shape[0] == 0
        # gfx1250 runs the fused cross-layer path for ALL sizes (prefill+decode);
        # there is no TileLang fallback available, so don't cap by M there.
        or (x.shape[0] > _FUSED_HC_POST_PRE_M_THRESHOLD and not _is_gfx1250_supported)
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

    from sglang.srt.models.deepseek_v4 import get_mhc_ops

    post_in = post.unsqueeze(-1) if post.ndim == 2 else post
    (
        residual,
        post_out,
        comb_out,
        layer_input_out,
    ) = get_mhc_ops().mhc_fused_post_pre(
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


def hc_boundary(
    layer,
    x: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: Optional[torch.Tensor],
    comb: Optional[torch.Tensor],
    pre_prev: Optional[torch.Tensor],
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, "HcCoefficients"]:
    """Fused sublayer boundary (ROCm): apply the pending hc_post of x onto residual, collapse
    with pre_prev (copy 0 when None) and take the mixing statistics. Returns (new_residual, y,
    coefficients); the coefficients' reduce + sinkhorn is still pending (see HcCoefficients)."""
    new_residual, y, coefficients = hc_boundary_fused_deferred(
        x,
        residual,
        post,
        comb,
        pre_prev,
        hc_fn,
        hc_scale,
        hc_base,
        layer.hc_mult,
        layer.hc_sinkhorn_iters,
        layer.rms_norm_eps,
        layer.hc_eps,
    )
    if new_residual is None:
        new_residual = residual
    if y is None:
        y = new_residual[:, 0, :].contiguous()
    return new_residual, y, coefficients


class HcMixes:
    """A boundary's pre / post / comb already computed (Mega-mHC), in the
    HcCoefficients interface the layer reads them through."""

    materialized = True

    def __init__(self, pre: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        self.pre, self.post, self.comb = pre, post, comb
        self.num_rows = pre.shape[0]

    def materialize(self) -> None:
        pass

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pre, self.post, self.comb


def can_use_mega_mhc(
    layer, residual: torch.Tensor, x: Optional[torch.Tensor], hc_fn: torch.Tensor
) -> bool:
    """Whether a fused boundary (plus its norm) can run as one FlyDSL Mega-mHC
    launch: gfx950, the fake-quant norm path, hc_mult 4, and dense bf16 rows."""
    if not (
        _is_gfx95_supported
        and envs.SGLANG_HIP_DSV41_MEGA_MHC.get()
        and layer.fused_rmsnorm_fake_quant
        and residual.dim() == 3
        and residual.shape[1] == 4
        and 1 <= residual.shape[0] <= envs.SGLANG_HIP_DSV41_MEGA_MHC_MAX_ROWS.get()
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and hc_fn.dtype == torch.float32
        and hc_fn.is_contiguous()
    ):
        return False
    return x is None or (
        x.shape == (residual.shape[0], residual.shape[2])
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def _mega_mhc(
    layer,
    x: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: Optional[torch.Tensor],
    comb: Optional[torch.Tensor],
    pre_prev: Optional[torch.Tensor],
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm,
    out_dtype: str,
):
    from aiter.ops.flydsl import flydsl_mega_mhc

    has_post = x is not None
    new_residual, post_out, comb_out, layer_input, next_pre = flydsl_mega_mhc(
        residual,
        hc_fn,
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        layer.rms_norm_eps,
        layer.hc_eps,
        layer.hc_eps,
        2.0,
        layer.hc_sinkhorn_iters,
        pre_prev.contiguous() if pre_prev is not None else None,
        x,
        post.contiguous().float() if has_post else None,
        comb.contiguous().float() if has_post else None,
        norm.weight.data,
        norm.variance_epsilon,
        residual_out=torch.empty_like(residual) if has_post else None,
        out_dtype=out_dtype,
    )
    mixes = HcMixes(next_pre, post_out.view(-1, 4), comb_out)
    return new_residual, layer_input, mixes


def mega_hc_boundary_input_norm(
    layer, x, residual, post, comb, pre_prev, hc_fn, hc_scale, hc_base
):
    """hc_boundary + layer._input_norm in one launch; returns (new_residual,
    (norm, x_quant), coefficients)."""
    if not layer._wqkv_a_native_consumer_checked:
        layer._wqkv_a_native_consumer = _mxfp8_consumer(
            layer.self_attn.wqkv_a if layer.self_attn.fuse_wqa_wkv else None
        )
        layer._wqkv_a_native_consumer_checked = True
    emit_fp8 = layer._wqkv_a_native_consumer
    new_residual, out, mixes = _mega_mhc(
        layer,
        x,
        residual,
        post,
        comb,
        pre_prev,
        hc_fn,
        hc_scale,
        hc_base,
        layer.input_layernorm,
        "mxfp8" if emit_fp8 else "fp8_grid",
    )
    if emit_fp8:
        norm, q, scale = out
        x_quant = Mxfp8Activation(q, scale)
    else:
        norm, grid = out
        x_quant = Fp8GridActivation(grid)
    return new_residual, (norm, x_quant), mixes


def mega_hc_boundary_post_attention_norm(
    layer, x, residual, post, comb, pre_prev, hc_fn, hc_scale, hc_base
):
    """hc_boundary + post_attention_norm in one launch; returns (new_residual,
    norm, coefficients), the norm carrying the shared expert's MXFP8 operand
    where post_attention_norm would attach one."""
    emit_fp8 = _ffn_norm_emits_mxfp8(layer, residual.shape[0])
    new_residual, out, mixes = _mega_mhc(
        layer,
        x,
        residual,
        post,
        comb,
        pre_prev,
        hc_fn,
        hc_scale,
        hc_base,
        layer.post_attention_layernorm,
        "mxfp8" if emit_fp8 else "bf16",
    )
    if emit_fp8:
        norm, q, scale = out
        norm._hip_mxfp8_operand = Mxfp8Activation(q, scale)
    else:
        norm = out
    return new_residual, norm, mixes


def _can_fuse_mhc(layer, residual: torch.Tensor, forward_batch) -> bool:
    return (
        _is_gfx95_supported
        and 1 <= residual.shape[0] <= ALL_REDUCE_MHC_MAX_ROWS
        and residual.shape[1:] == (4, MHC_HIDDEN_SIZE)
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and layer.config.model_type == "deepseek_v41"
        and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        and get_parallel().attn_dp_size == 1
        and get_parallel().tp_size == 4
        and not layer.dsa_enable_prefill_cp
        and not get_forward().sp_active
        and not is_batch_invariant_mode_enabled()
        and not get_exec().deterministic.enable_deterministic_inference
    )


def _make_mhc_fusion(
    residual: torch.Tensor, coefficients: "HcCoefficients", comm
) -> Optional[MhcPostOperands]:
    # comm is the group's aiter CustomAllreduce, or None when custom all-reduce is off
    if comm is None or comm.disabled or not comm.enable_register_for_capturing:
        return None
    coefficients.materialize()
    return MhcPostOperands(residual, coefficients.post, coefficients.comb)


def attention_mhc_fusion(layer, residual, coefficients, forward_batch):
    if not (
        _can_fuse_mhc(layer, residual, forward_batch)
        and layer.self_attn.attn_tp_size == 4
        and layer.self_attn.wo_b.reduce_results
    ):
        return None
    return _make_mhc_fusion(
        residual, coefficients, get_parallel().attn_tp_group.ca_comm
    )


def moe_mhc_fusion(layer, residual, coefficients, forward_batch):
    if not (
        _can_fuse_mhc(layer, residual, forward_batch)
        and layer.mlp.tp_size == 4
        and not layer.mlp._shared_expert_tp1
        and get_moe_a2a_backend().is_none()
        # A reduce some later step already owns, or one the combine performed, is
        # not ours to fuse into the post.
        and not should_skip_post_experts_all_reduce(is_tp_path=True)
    ):
        return None
    return _make_mhc_fusion(residual, coefficients, get_parallel().tp_group.ca_comm)


def apply_attention_mhc(x: torch.Tensor, state: MhcPostOperands) -> torch.Tensor:
    """Attention's all-reduce fused with hc_post; returns the rebuilt streams."""
    return all_reduce_mhc_post(
        x, state.residual, state.post, state.comb, get_parallel().attn_tp_group.ca_comm
    )


def apply_moe_mhc(x: torch.Tensor, state: MhcPostOperands) -> torch.Tensor:
    """The same for the MoE's post-experts reduce, on rows it merged but left
    unreduced."""
    return all_reduce_mhc_post(
        x, state.residual, state.post, state.comb, get_parallel().tp_group.ca_comm
    )


def _gfx95_dense_post_attention_norm(layer, x: torch.Tensor, coefficients):
    """layer.post_attention_layernorm(x) hosting the pending reduce + sinkhorn where
    the gfx950 norm kernel is available; the module norm plus a standalone reduce +
    sinkhorn elsewhere."""
    if _is_gfx95_supported:
        return post_attention_norm(layer, x, coefficients)
    coefficients.materialize()
    return layer.post_attention_layernorm(x)
