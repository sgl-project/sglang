"""Qwen3.5 GDN gated RMSNorm + FP8 per-token quant folded into ``out_proj``, on ROCm.

A GDN layer ends by normalizing ``core_attn_out`` against the ``z`` gate, quantizing per
token, then running ``out_proj`` as an a8w8 GEMM. AITER does the first three in one
kernel and hands the quantized activation straight to the GEMM.

:func:`owns_out_proj` returns False -- leaving the tail in ``qwen3_5`` untouched -- off
ROCm, without the AITER kernel, under ``SGLANG_DISABLE_GDN_OUT_PROJ_FUSION``, or on an
``out_proj`` that is not per-token a8w8 in the shapes the kernel handles.
"""

from __future__ import annotations

import torch

from sglang.srt.utils import get_bool_env_var, is_hip

IS_HIP = is_hip()

gated_rmsnorm_fp8_per_token_quant = None
if IS_HIP and get_bool_env_var("SGLANG_USE_AITER"):
    try:
        from aiter import gemm_a8w8_bpreshuffle
        from aiter.ops.gated_rmsnorm_fp8_per_token_quant import (
            gated_rmsnorm_fp8_per_token_quant,
        )
    except Exception:
        gated_rmsnorm_fp8_per_token_quant = None

ENABLED = gated_rmsnorm_fp8_per_token_quant is not None and not get_bool_env_var(
    "SGLANG_DISABLE_GDN_OUT_PROJ_FUSION", "False"
)

_OWNS_ATTR = "_gdn_out_proj_fused"


def _eligible(layer: torch.nn.Module) -> bool:
    out_proj = layer.out_proj
    weight = getattr(out_proj, "weight", None)
    weight_scale = getattr(out_proj, "weight_scale", None)
    # Either quant path sets this once it holds the pre-shuffled weight the GEMM wants.
    per_token_a8w8 = bool(
        getattr(
            getattr(out_proj, "quant_method", None), "use_per_token_if_dynamic", False
        )
        or getattr(getattr(out_proj, "scheme", None), "per_token", False)
    )
    return (
        weight is not None
        and weight_scale is not None
        # AttnFP8 weights are e4m3fn; the fused kernels have no fnuz path.
        and weight.dtype == torch.float8_e4m3fn
        and per_token_a8w8
        and layer.head_v_dim == 128
        and (layer.num_v_heads // layer.attn_tp_size) <= 128
        and layer.output_gate_type in (None, "silu")
    )


def owns_out_proj(layer: torch.nn.Module) -> bool:
    """Whether ``layer`` runs its norm, gate, quant and ``out_proj`` as one fused tail.

    Settled per layer on the first forward: a quant method only advertises per-token
    a8w8 once its weights are loaded.
    """
    owns = getattr(layer, _OWNS_ATTR, None)
    if owns is None:
        owns = ENABLED and _eligible(layer)
        setattr(layer, _OWNS_ATTR, owns)
    return owns


def apply(
    layer: torch.nn.Module,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    z_shape_og: torch.Size,
) -> torch.Tensor:
    """``layer``'s ``out_proj`` over a gated-RMSNorm'd, FP8 per-token quantized input."""
    num_tokens, num_heads, head_dim = z_shape_og
    out_proj = layer.out_proj

    qinput = torch.empty(
        (num_tokens, num_heads * head_dim),
        dtype=out_proj.weight.dtype,
        device=core_attn_out.device,
    )
    x_scale = torch.empty(
        (num_tokens,), dtype=torch.float32, device=core_attn_out.device
    )
    gated_rmsnorm_fp8_per_token_quant(
        qinput,
        x_scale,
        core_attn_out.reshape(z_shape_og),
        z.reshape(z_shape_og),
        layer.norm.weight,
        layer.layer_norm_epsilon,
    )

    # The AITER per-token branch of apply_fp8_linear, minus the quant just done.
    # WQ wants (n, k); out_proj.weight is (k, n).
    output = gemm_a8w8_bpreshuffle(
        XQ=qinput,
        WQ=out_proj.weight.T,
        x_scale=x_scale.view(-1, 1),
        w_scale=out_proj.weight_scale,
        dtype=torch.bfloat16,
    )
    if type(output) is tuple and len(output) == 2:
        output = output[0]
    return torch.narrow(output, 0, 0, num_tokens).view(
        num_tokens, out_proj.weight.shape[1]
    )
