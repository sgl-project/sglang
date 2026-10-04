"""Cake (FlashInfer) backends for the ``diffusion`` operator group: MiniMax-H3.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels`, which import FlashInfer only
when a kernel is actually called. Each ``cake_<name>`` wrapper is the explicit
Cake entry; callers gate on the adapter's ``supports_<name>``.

Op ids follow the FlashInfer entry names. SM100/SM103 (tcgen05) and SM120
(``mma.sync``) variants are separate FlashInfer entries with non-interchangeable
prepared-weight layouts, so they carry separate op ids and capability windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_ADAPTERS = "sglang.kernels.cake_kernels."
_PRE = _ADAPTERS + "diffusion_minimax_h3_pre_attention:"
_PROJ = _ADAPTERS + "diffusion_minimax_h3_proj:"
_ATTN = _ADAPTERS + "diffusion_minimax_h3_attention:"
_SM120 = _ADAPTERS + "diffusion_minimax_h3_sm120:"

_CUDA = frozenset({CapabilityRequirement.CUDA})
_SM100_103 = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})
_SM120_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 0))})
_SM120_121 = frozenset({CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 1))})
_SM100_103_OR_SM120 = frozenset(
    {
        CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3)),
        CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 0)),
    }
)
_SM90_103_OR_SM120_121 = frozenset(
    {
        CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(10, 3)),
        CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 1)),
    }
)

# (op name, target, capabilities, supported dtypes, in_place, format, description)
_SPECS: Tuple[Tuple[str, str, frozenset, Tuple[str, ...], bool, str, str], ...] = (
    # --- SM100/103 pre-attention -------------------------------------------
    (
        "minimax_h3_bf16_pre_attention",
        _PRE + "minimax_h3_bf16_pre_attention",
        _SM100_103,
        ("bfloat16",),
        True,
        "BF16 x [M,5376], AdaLN tables [rows,5376] (any rows, strided ok), int64 "
        "index, (rope_cos_sin [S,96], rope_positions int64 [M]), eps/qk_eps -> "
        "caller-owned out [P,M,56/P,3,128]; P in {1,2,4,8}",
        "Cake MiniMax-H3 fused RMSNorm+AdaLN+BF16 QKV+QK-norm+RoPE+pack (SM100/103).",
    ),
    (
        "prepare_minimax_h3_mxfp8_pre_attention",
        _PRE + "prepare_minimax_h3_mxfp8_pre_attention",
        _SM100_103,
        ("bfloat16", "float8_e4m3fn", "uint8"),
        True,
        "prepare -> runner() -> (out_q e4m3 [P,M,56/P,3,128], out_sf uint8); "
        "exact (M,P) route table",
        "Cake MiniMax-H3 MXFP8 pre-attention prepared runner (SM100/103).",
    ),
    (
        "prepare_minimax_h3_nvfp4_pre_attention",
        _PRE + "prepare_minimax_h3_nvfp4_pre_attention",
        _SM100_103,
        ("bfloat16", "uint8", "float32"),
        True,
        "prepare -> runner() -> (out_q uint8 [P,M,56/P,3,64], out_sf uint8); "
        "fused tcgen05 NVFP4 QKV GEMM",
        "Cake MiniMax-H3 NVFP4 (W4A4) pre-attention prepared runner (SM100/103).",
    ),
    (
        "prepare_minimax_h3_qkv_quantize_pack",
        _PRE + "prepare_minimax_h3_qkv_quantize_pack",
        _SM100_103,
        ("bfloat16", "uint8", "float8_e4m3fn"),
        True,
        "prepare -> runner() -> (out_q, out_sf) Ulysses send buffer; "
        "format in {nvfp4, mxfp8}",
        "Cake MiniMax-H3 QKV quantize-and-pack prepared runner (SM100/103).",
    ),
    (
        "minimax_h3_qkv_quantize_pack",
        _PRE + "minimax_h3_qkv_quantize_pack",
        _SM100_103,
        ("bfloat16", "uint8", "float8_e4m3fn"),
        False,
        "one-shot (q,k,v [M,56,128], P, format) -> (out_q, out_sf); allocates, "
        "not CUDA-graph safe",
        "Cake MiniMax-H3 QKV quantize-and-pack, one-shot (SM100/103).",
    ),
    # --- SM100/103 FC1 + SwiGLU ---------------------------------------------
    (
        "minimax_h3_fc1_swiglu",
        _PROJ + "minimax_h3_fc1_swiglu",
        _SM100_103,
        ("bfloat16",),
        False,
        "BF16 x [M,5376] + AdaLN tables [rows,5376] (any rows, strided ok) + int64 "
        "index + fc1_weight [28672,5376] -> out [M,14336]",
        "Cake MiniMax-H3 BF16 norm+AdaLN+FC1+SwiGLU (SM100/103 tcgen05).",
    ),
    (
        "minimax_h3_fc1_swiglu_mxfp8",
        _PROJ + "minimax_h3_fc1_swiglu_mxfp8",
        _SM100_103,
        ("bfloat16", "float8_e4m3fn", "uint8"),
        False,
        "MXFP8 weight tiles (prepare_minimax_h3_fc1_weight_mxfp8) -> out [M,14336]",
        "Cake MiniMax-H3 MXFP8 FC1+SwiGLU (SM100/103 tcgen05).",
    ),
    (
        "minimax_h3_fc1_swiglu_nvfp4",
        _PROJ + "minimax_h3_fc1_swiglu_nvfp4",
        _SM100_103_OR_SM120,
        ("bfloat16", "uint8", "float32"),
        False,
        "NVFP4 weight tiles -> out [M,14336]; arch dispatcher (tcgen05 on 10.x, "
        "mma.sync on 12.0)",
        "Cake MiniMax-H3 NVFP4 FC1+SwiGLU (SM100/103; SM120 route on cc 12.0).",
    ),
    (
        "prepare_minimax_h3_fc1_weight_mxfp8",
        _PROJ + "prepare_minimax_h3_fc1_weight_mxfp8",
        _CUDA,
        ("bfloat16",),
        False,
        "fc1_weight bf16 [28672,5376] -> (e4m3, combined 256-row scale tiles); "
        "SM100/103 layout",
        "Cake MiniMax-H3 FC1 MXFP8 weight preparation (SM100/103 layout).",
    ),
    (
        "prepare_minimax_h3_fc1_weight_nvfp4",
        _PROJ + "prepare_minimax_h3_fc1_weight_nvfp4",
        _CUDA,
        ("bfloat16",),
        False,
        "fc1_weight bf16 -> (uint8 e2m1 pairs, scale tiles); layout follows the "
        "weight's device arch",
        "Cake MiniMax-H3 FC1 NVFP4 weight preparation (arch dispatcher).",
    ),
    # --- SM100/103 out-projection ------------------------------------------
    (
        "minimax_h3_out_proj",
        _PROJ + "minimax_h3_out_proj",
        _SM100_103,
        ("bfloat16",),
        False,
        "attn_out [P,M,56/P,128] receive layout + o_weight [5376,7168] + gate "
        "[rows,5376] (any rows, strided ok) + int64 gate_index -> "
        "bf16(residual + bf16(gate * bf16(A@W^T)))",
        "Cake MiniMax-H3 BF16 gated-residual out-projection (SM100/103 tcgen05).",
    ),
    (
        "minimax_h3_out_proj_mxfp8",
        _PROJ + "minimax_h3_out_proj_mxfp8",
        _SM100_103,
        ("bfloat16", "float8_e4m3fn", "uint8"),
        False,
        "MXFP8 weight tiles (prepare_minimax_h3_o_weight_mxfp8) -> out [M,5376]",
        "Cake MiniMax-H3 MXFP8 gated-residual out-projection (SM100/103).",
    ),
    (
        "minimax_h3_out_proj_nvfp4",
        _PROJ + "minimax_h3_out_proj_nvfp4",
        _SM100_103,
        ("bfloat16", "uint8", "float32"),
        False,
        "NVFP4 weight tiles (prepare_minimax_h3_o_weight_nvfp4) -> out [M,5376]",
        "Cake MiniMax-H3 NVFP4 gated-residual out-projection (SM100/103).",
    ),
    (
        "prepare_minimax_h3_o_weight_mxfp8",
        _PROJ + "prepare_minimax_h3_o_weight_mxfp8",
        _CUDA,
        ("bfloat16",),
        False,
        "o_weight bf16 [5376,7168] -> (e4m3, combined scale tiles); SM100/103 layout",
        "Cake MiniMax-H3 out-proj MXFP8 weight preparation (SM100/103 layout).",
    ),
    (
        "prepare_minimax_h3_o_weight_nvfp4",
        _PROJ + "prepare_minimax_h3_o_weight_nvfp4",
        _CUDA,
        ("bfloat16",),
        False,
        "o_weight bf16 [5376,7168] -> (uint8 e2m1 pairs, scale tiles); SM100/103 layout",
        "Cake MiniMax-H3 out-proj NVFP4 weight preparation (SM100/103 layout).",
    ),
    # --- SM100/103 packed-varlen attention ----------------------------------
    (
        "minimax_h3_varlen_attention",
        _ATTN + "minimax_h3_varlen_attention",
        _SM100_103,
        ("bfloat16",),
        False,
        "BF16 THD q,k,v [T,H,128] (strided token-major views ok) + int32 "
        "cu_seqlens [B+1] -> contiguous out [T,H,128]; noncausal, H_q == H_kv",
        "Cake MiniMax-H3 packed-varlen BF16 attention (SM100/103).",
    ),
    (
        "minimax_h3_varlen_nvfp4_attention",
        _ATTN + "minimax_h3_varlen_nvfp4_attention",
        _SM100_103,
        ("bfloat16",),
        False,
        "BF16 THD inputs; NVFP4 QK with E4M3 (pv_mode=fp8) or NVFP4 (fp4) PV",
        "Cake MiniMax-H3 packed-varlen NVFP4 attention (SM100/103).",
    ),
    (
        "prepare_minimax_h3_varlen_attention",
        _ATTN + "prepare_minimax_h3_varlen_attention",
        _SM100_103,
        ("bfloat16",),
        True,
        "prepare -> runner.launch() writes runner.out; bound to one cu_seqlens",
        "Cake MiniMax-H3 packed-varlen BF16 attention prepared runner (SM100/103).",
    ),
    (
        "prepare_minimax_h3_varlen_nvfp4_attention",
        _ATTN + "prepare_minimax_h3_varlen_nvfp4_attention",
        _SM100_103,
        ("bfloat16",),
        True,
        "prepare -> runner.launch() (quantize + attention); bound to one cu_seqlens",
        "Cake MiniMax-H3 packed-varlen NVFP4 attention prepared runner (SM100/103).",
    ),
    # --- dense attention -----------------------------------------------------
    (
        "minimax_h3_dense_attention",
        _ATTN + "minimax_h3_dense_attention",
        _SM90_103_OR_SM120_121,
        ("bfloat16",),
        False,
        "BF16 q,k,v [tokens,7168] -> out; 56 heads x 128, batch 1, tokens <= 131072",
        "Cake MiniMax-H3 dense BF16 attention (SM120 target; builds for 9.x/10.x/12.x).",
    ),
    # --- SM120 quantized packed-varlen attention ----------------------------
    (
        "minimax_h3_sm120_varlen_attention_fp8",
        _ATTN + "minimax_h3_sm120_varlen_attention_fp8",
        _SM120_121,
        ("bfloat16",),
        False,
        "BF16 THD q,k,v [T,H,128] + int32 cu_seqlens -> out; FP8 mma.sync operands",
        "Cake MiniMax-H3 SM120 FP8 packed-varlen attention.",
    ),
    (
        "minimax_h3_sm120_varlen_attention_nvfp4",
        _ATTN + "minimax_h3_sm120_varlen_attention_nvfp4",
        _SM120_121,
        ("bfloat16",),
        False,
        "BF16 THD q,k,v [T,H,128] + int32 cu_seqlens -> out; NVFP4 mma.sync operands",
        "Cake MiniMax-H3 SM120 NVFP4 packed-varlen attention (experimental).",
    ),
    # --- SM120 quantized pre-attention --------------------------------------
    (
        "minimax_h3_fp8_pre_attention",
        _SM120 + "minimax_h3_fp8_pre_attention",
        _SM120_ONLY,
        ("bfloat16", "float8_e4m3fn", "float32"),
        False,
        "W8A8 x [M,5376] -> (q,k,v) per-kind [M,56,*]; out_mode bf16/e4m3/nvfp4",
        "Cake MiniMax-H3 SM120 FP8 pre-attention (mma.sync).",
    ),
    (
        "minimax_h3_nvfp4_pre_attention",
        _SM120 + "minimax_h3_nvfp4_pre_attention",
        _SM120_ONLY,
        ("bfloat16", "uint8", "float32"),
        False,
        "W4A4 x [M,5376] -> (q,k,v) per-kind [M,56,*]; pass alpha to avoid a host sync",
        "Cake MiniMax-H3 SM120 NVFP4 pre-attention (mma.sync).",
    ),
    (
        "quantize_minimax_h3_qkv_weight_fp8",
        _SM120 + "quantize_minimax_h3_qkv_weight_fp8",
        _CUDA,
        ("bfloat16",),
        False,
        "qkv_weight [21504,5376] -> (e4m3, f32 [21504] per-row scales); plain row order",
        "Cake MiniMax-H3 QKV FP8 weight quantization (SM120 pre-attention).",
    ),
    (
        "quantize_minimax_h3_qkv_weight_nvfp4",
        _SM120 + "quantize_minimax_h3_qkv_weight_nvfp4",
        _CUDA,
        ("bfloat16",),
        False,
        "qkv_weight [21504,5376] -> (uint8 e2m1 pairs, swizzled scales, f32 [1] global scale)",
        "Cake MiniMax-H3 QKV NVFP4 weight quantization (SM120 pre-attention).",
    ),
    # --- SM120 FC1 + SwiGLU --------------------------------------------------
    (
        "minimax_h3_fc1_swiglu_fp8",
        _SM120 + "minimax_h3_fc1_swiglu_fp8",
        _SM120_ONLY,
        ("bfloat16", "float8_e4m3fn", "float32"),
        False,
        "W8A8 x [M,5376] + SM120-interleaved fc1 weight -> out [M,14336]",
        "Cake MiniMax-H3 SM120 FP8 FC1+SwiGLU (mma.sync).",
    ),
    (
        "prepare_minimax_h3_fc1_weight_fp8",
        _SM120 + "prepare_minimax_h3_fc1_weight_fp8",
        _CUDA,
        ("bfloat16",),
        False,
        "fc1_weight bf16 [28672,5376] -> (e4m3, f32 [28672]) in SM120 interleaved row order",
        "Cake MiniMax-H3 FC1 FP8 weight preparation (SM120 layout).",
    ),
    (
        "prepare_minimax_h3_fc1_weight_nvfp4_sm120",
        _SM120 + "prepare_minimax_h3_fc1_weight_nvfp4_sm120",
        _CUDA,
        ("bfloat16",),
        False,
        "fc1_weight bf16 + w_global_scale -> (uint8 [28672,2688], uint8 [28672*336]) SM120 layout",
        "Cake MiniMax-H3 FC1 NVFP4 weight preparation (SM120 layout).",
    ),
    # --- SM120 out-projection -----------------------------------------------
    (
        "minimax_h3_fp8_out_proj",
        _SM120 + "minimax_h3_fp8_out_proj",
        _SM120_ONLY,
        ("bfloat16", "float8_e4m3fn", "float32"),
        False,
        "attn_out [M,7168] NHD row-major + FP8 o_weight -> bf16(residual + bf16(gate*o))",
        "Cake MiniMax-H3 SM120 FP8 gated-residual out-projection (one launch).",
    ),
    (
        "minimax_h3_nvfp4_out_proj",
        _SM120 + "minimax_h3_nvfp4_out_proj",
        _SM120_ONLY,
        ("bfloat16", "uint8", "float32"),
        False,
        "attn_out [M,7168] + NVFP4 o_weight -> out [M,5376]; host .item() per call "
        "(not CUDA-graph safe)",
        "Cake MiniMax-H3 SM120 NVFP4 gated-residual out-projection.",
    ),
    (
        "quantize_minimax_h3_o_weight_fp8",
        _SM120 + "quantize_minimax_h3_o_weight_fp8",
        _CUDA,
        ("bfloat16",),
        False,
        "o_weight [5376,7168] -> (e4m3, f32 [5376] per-output-channel scales)",
        "Cake MiniMax-H3 out-proj FP8 weight quantization (SM120 layout).",
    ),
    (
        "quantize_minimax_h3_o_weight_nvfp4",
        _SM120 + "quantize_minimax_h3_o_weight_nvfp4",
        _CUDA,
        ("bfloat16",),
        False,
        "o_weight [5376,7168] -> (uint8 e2m1 pairs, swizzled scales, f32 [1] global scale)",
        "Cake MiniMax-H3 out-proj NVFP4 weight quantization (SM120 layout).",
    ),
)

for _name, _target, _caps, _dtypes, _in_place, _format, _description in _SPECS:
    register_kernel(
        KernelSpec(
            op=f"diffusion.{_name}",
            backend=KernelBackend.FLASHINFER,
            target=_target,
            capabilities=_caps,
            format_signature=FormatSignature(
                supported_dtypes=_dtypes, in_place=_in_place, description=_format
            ),
            description=_description + " Distributed by FlashInfer.",
        )
    )
del _name, _target, _caps, _dtypes, _in_place, _format, _description


def _k(name: str):
    return get_kernel(f"diffusion.{name}", KernelBackend.FLASHINFER)


Scalar = Union["torch.Tensor", float]


# ---------------------------------------------------------------------------
# SM100/103 pre-attention
# ---------------------------------------------------------------------------


def cake_minimax_h3_bf16_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    ulysses_degree: int,
    out: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_bf16_pre_attention")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        ulysses_degree=ulysses_degree,
        out=out,
        eps=eps,
    )


def cake_prepare_minimax_h3_mxfp8_pre_attention(
    *,
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    activation_q: torch.Tensor,
    activation_sf: torch.Tensor,
    qkv_bf16: torch.Tensor,
    gemm_workspace: torch.Tensor,
    P: int,
    debug_q_bf16: Optional[torch.Tensor] = None,
    debug_k_bf16: Optional[torch.Tensor] = None,
    eps: float = 1.0e-5,
) -> Any:
    """Explicit Cake entry point; returns the FlashInfer prepared runner."""
    return _k("prepare_minimax_h3_mxfp8_pre_attention")(
        x=x,
        x_norm_weight=x_norm_weight,
        adaln_scale=adaln_scale,
        adaln_shift=adaln_shift,
        adaln_index=adaln_index,
        qkv_weight_q=qkv_weight_q,
        qkv_weight_sf=qkv_weight_sf,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        rope_cos_sin=rope_cos_sin,
        out_q=out_q,
        out_sf=out_sf,
        activation_q=activation_q,
        activation_sf=activation_sf,
        qkv_bf16=qkv_bf16,
        gemm_workspace=gemm_workspace,
        P=P,
        debug_q_bf16=debug_q_bf16,
        debug_k_bf16=debug_k_bf16,
        eps=eps,
    )


def cake_prepare_minimax_h3_nvfp4_pre_attention(
    *,
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    x_global_scale: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    w_global_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    out_global_scale: torch.Tensor,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    activation_q: torch.Tensor,
    activation_sf: torch.Tensor,
    P: int,
    debug_q_bf16: Optional[torch.Tensor] = None,
    debug_k_bf16: Optional[torch.Tensor] = None,
    debug_adaln_bf16: Optional[torch.Tensor] = None,
    eps: float = 1.0e-5,
) -> Any:
    """Explicit Cake entry point; returns the FlashInfer prepared runner."""
    return _k("prepare_minimax_h3_nvfp4_pre_attention")(
        x=x,
        x_norm_weight=x_norm_weight,
        adaln_scale=adaln_scale,
        adaln_shift=adaln_shift,
        adaln_index=adaln_index,
        x_global_scale=x_global_scale,
        qkv_weight_q=qkv_weight_q,
        qkv_weight_sf=qkv_weight_sf,
        w_global_scale=w_global_scale,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        rope_cos_sin=rope_cos_sin,
        out_global_scale=out_global_scale,
        out_q=out_q,
        out_sf=out_sf,
        activation_q=activation_q,
        activation_sf=activation_sf,
        P=P,
        debug_q_bf16=debug_q_bf16,
        debug_k_bf16=debug_k_bf16,
        debug_adaln_bf16=debug_adaln_bf16,
        eps=eps,
    )


def cake_prepare_minimax_h3_qkv_quantize_pack(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    P: int,
    format: str,
    out_global_scale: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns the FlashInfer prepared runner."""
    return _k("prepare_minimax_h3_qkv_quantize_pack")(
        q=q,
        k=k,
        v=v,
        out_q=out_q,
        out_sf=out_sf,
        P=P,
        format=format,
        out_global_scale=out_global_scale,
    )


def cake_minimax_h3_qkv_quantize_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    P: int,
    format: str,
    out_global_scale: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (one-shot; not CUDA-graph safe)."""
    return _k("minimax_h3_qkv_quantize_pack")(
        q,
        k,
        v,
        P,
        format,
        out_global_scale=out_global_scale,
        out_q=out_q,
        out_sf=out_sf,
    )


# ---------------------------------------------------------------------------
# SM100/103 FC1 + SwiGLU
# ---------------------------------------------------------------------------


def cake_minimax_h3_fc1_swiglu(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_fc1_swiglu")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight,
        out=out,
        workspace=workspace,
        eps=eps,
    )


def cake_minimax_h3_fc1_swiglu_mxfp8(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_fc1_swiglu_mxfp8")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_scale_tiles,
        out=out,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
        eps=eps,
    )


def cake_minimax_h3_fc1_swiglu_nvfp4(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_fc1_swiglu_nvfp4")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        a_global_scale,
        fc1_weight_q,
        fc1_scale_tiles,
        alpha,
        out=out,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
        eps=eps,
    )


def cake_prepare_minimax_h3_fc1_weight_mxfp8(
    fc1_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight preparation, SM100/103 layout)."""
    return _k("prepare_minimax_h3_fc1_weight_mxfp8")(fc1_weight)


def cake_prepare_minimax_h3_fc1_weight_nvfp4(
    fc1_weight: torch.Tensor, w_global_scale: Union[torch.Tensor, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight preparation, arch dispatcher)."""
    return _k("prepare_minimax_h3_fc1_weight_nvfp4")(fc1_weight, w_global_scale)


# ---------------------------------------------------------------------------
# SM100/103 out-projection
# ---------------------------------------------------------------------------


def cake_minimax_h3_out_proj(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_out_proj")(
        attn_out, o_weight, gate, gate_index, residual, out=out
    )


def cake_minimax_h3_out_proj_mxfp8(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_out_proj_mxfp8")(
        attn_out,
        o_weight_q,
        o_scale_tiles,
        gate,
        gate_index,
        residual,
        out=out,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
    )


def cake_minimax_h3_out_proj_nvfp4(
    attn_out: torch.Tensor,
    a_global_scale: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_scale_tiles: torch.Tensor,
    alpha: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_out_proj_nvfp4")(
        attn_out,
        a_global_scale,
        o_weight_q,
        o_scale_tiles,
        alpha,
        gate,
        gate_index,
        residual,
        out=out,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
    )


def cake_prepare_minimax_h3_o_weight_mxfp8(
    o_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight preparation, SM100/103 layout)."""
    return _k("prepare_minimax_h3_o_weight_mxfp8")(o_weight)


def cake_prepare_minimax_h3_o_weight_nvfp4(
    o_weight: torch.Tensor, w_global_scale: Union[torch.Tensor, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight preparation, SM100/103 layout)."""
    return _k("prepare_minimax_h3_o_weight_nvfp4")(o_weight, w_global_scale)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def cake_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_varlen_attention")(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
    )


def cake_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_varlen_nvfp4_attention")(
        query,
        key,
        value,
        cu_seqlens,
        pv_mode=pv_mode,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
    )


def cake_prepare_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> Any:
    """Explicit Cake entry point; returns the FlashInfer prepared runner."""
    return _k("prepare_minimax_h3_varlen_attention")(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
    )


def cake_prepare_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    workspace: Optional[Dict[str, torch.Tensor]] = None,
) -> Any:
    """Explicit Cake entry point; returns the FlashInfer prepared runner."""
    return _k("prepare_minimax_h3_varlen_nvfp4_attention")(
        query,
        key,
        value,
        cu_seqlens,
        pv_mode=pv_mode,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        workspace=workspace,
    )


def cake_minimax_h3_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_dense_attention")(q, k, v, out=out)


def cake_minimax_h3_sm120_varlen_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_sm120_varlen_attention_fp8")(
        q,
        k,
        v,
        cu_seqlens,
        out,
        cu_seqlens_host=cu_seqlens_host,
        softmax_scale=softmax_scale,
    )


def cake_minimax_h3_sm120_varlen_attention_nvfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_sm120_varlen_attention_nvfp4")(
        q,
        k,
        v,
        cu_seqlens,
        out,
        cu_seqlens_host=cu_seqlens_host,
        softmax_scale=softmax_scale,
    )


# ---------------------------------------------------------------------------
# SM120 quantized pre-attention / FC1 / out-projection
# ---------------------------------------------------------------------------


def cake_minimax_h3_fp8_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    eps: float = 1.0e-5,
    out_mode: str = "bf16",
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    q_sf: Optional[torch.Tensor] = None,
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
    q_descale: Optional[Scalar] = None,
    k_descale: Optional[Scalar] = None,
    v_descale: Optional[Scalar] = None,
    q_global_scale: Optional[Scalar] = None,
    k_global_scale: Optional[Scalar] = None,
    v_global_scale: Optional[Scalar] = None,
    act_q: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns ``MiniMaxH3PreAttentionOutput``."""
    return _k("minimax_h3_fp8_pre_attention")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight_q,
        qkv_weight_scale,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        eps=eps,
        out_mode=out_mode,
        q=q,
        k=k,
        v=v,
        q_sf=q_sf,
        k_sf=k_sf,
        v_sf=v_sf,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        q_global_scale=q_global_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        act_q=act_q,
        act_scale=act_scale,
    )


def cake_minimax_h3_nvfp4_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    qkv_weight_global_scale: Scalar,
    act_global_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    eps: float = 1.0e-5,
    out_mode: str = "bf16",
    alpha: Optional[float] = None,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    q_sf: Optional[torch.Tensor] = None,
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
    q_descale: Optional[Scalar] = None,
    k_descale: Optional[Scalar] = None,
    v_descale: Optional[Scalar] = None,
    q_global_scale: Optional[Scalar] = None,
    k_global_scale: Optional[Scalar] = None,
    v_global_scale: Optional[Scalar] = None,
    act_q: Optional[torch.Tensor] = None,
    act_sf: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns ``MiniMaxH3PreAttentionOutput``."""
    return _k("minimax_h3_nvfp4_pre_attention")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight_q,
        qkv_weight_sf,
        qkv_weight_global_scale,
        act_global_scale,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        eps=eps,
        out_mode=out_mode,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        q_sf=q_sf,
        k_sf=k_sf,
        v_sf=v_sf,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        q_global_scale=q_global_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        act_q=act_q,
        act_sf=act_sf,
    )


def cake_quantize_minimax_h3_qkv_weight_fp8(
    qkv_weight: torch.Tensor, chunk_rows: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight quantization, SM120 layout)."""
    return _k("quantize_minimax_h3_qkv_weight_fp8")(qkv_weight, chunk_rows=chunk_rows)


def cake_quantize_minimax_h3_qkv_weight_nvfp4(
    qkv_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight quantization, SM120 layout)."""
    return _k("quantize_minimax_h3_qkv_weight_nvfp4")(qkv_weight)


def cake_minimax_h3_fc1_swiglu_fp8(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_scale: Optional[torch.Tensor] = None,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_fc1_swiglu_fp8")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        fc1_weight_q,
        fc1_weight_scale,
        out=out,
        workspace_q=workspace_q,
        workspace_scale=workspace_scale,
        eps=eps,
    )


def cake_prepare_minimax_h3_fc1_weight_fp8(
    fc1_weight: torch.Tensor, chunk_rows: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight preparation, SM120 layout)."""
    return _k("prepare_minimax_h3_fc1_weight_fp8")(fc1_weight, chunk_rows=chunk_rows)


def cake_prepare_minimax_h3_fc1_weight_nvfp4_sm120(
    fc1_weight: torch.Tensor, w_global_scale: Scalar
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight preparation, SM120 layout)."""
    return _k("prepare_minimax_h3_fc1_weight_nvfp4_sm120")(fc1_weight, w_global_scale)


def cake_minimax_h3_fp8_out_proj(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    act_q: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _k("minimax_h3_fp8_out_proj")(
        attn_out,
        o_weight_q,
        o_weight_scale,
        gate,
        gate_index,
        residual,
        out=out,
        act_q=act_q,
        act_scale=act_scale,
    )


def cake_minimax_h3_nvfp4_out_proj(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_sf: torch.Tensor,
    o_weight_global_scale: Scalar,
    act_global_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    act_q: Optional[torch.Tensor] = None,
    act_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point (host ``.item()`` per call; not CUDA-graph safe)."""
    return _k("minimax_h3_nvfp4_out_proj")(
        attn_out,
        o_weight_q,
        o_weight_sf,
        o_weight_global_scale,
        act_global_scale,
        gate,
        gate_index,
        residual,
        out=out,
        act_q=act_q,
        act_sf=act_sf,
    )


def cake_quantize_minimax_h3_o_weight_fp8(
    o_weight: torch.Tensor, chunk_rows: int = 1792
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight quantization, SM120 layout)."""
    return _k("quantize_minimax_h3_o_weight_fp8")(o_weight, chunk_rows=chunk_rows)


def cake_quantize_minimax_h3_o_weight_nvfp4(
    o_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point (offline weight quantization, SM120 layout)."""
    return _k("quantize_minimax_h3_o_weight_nvfp4")(o_weight)


__all__ = [
    "cake_minimax_h3_bf16_pre_attention",
    "cake_minimax_h3_dense_attention",
    "cake_minimax_h3_fc1_swiglu",
    "cake_minimax_h3_fc1_swiglu_fp8",
    "cake_minimax_h3_fc1_swiglu_mxfp8",
    "cake_minimax_h3_fc1_swiglu_nvfp4",
    "cake_minimax_h3_fp8_out_proj",
    "cake_minimax_h3_fp8_pre_attention",
    "cake_minimax_h3_nvfp4_out_proj",
    "cake_minimax_h3_nvfp4_pre_attention",
    "cake_minimax_h3_out_proj",
    "cake_minimax_h3_out_proj_mxfp8",
    "cake_minimax_h3_out_proj_nvfp4",
    "cake_minimax_h3_qkv_quantize_pack",
    "cake_minimax_h3_sm120_varlen_attention_fp8",
    "cake_minimax_h3_sm120_varlen_attention_nvfp4",
    "cake_minimax_h3_varlen_attention",
    "cake_minimax_h3_varlen_nvfp4_attention",
    "cake_prepare_minimax_h3_fc1_weight_fp8",
    "cake_prepare_minimax_h3_fc1_weight_mxfp8",
    "cake_prepare_minimax_h3_fc1_weight_nvfp4",
    "cake_prepare_minimax_h3_fc1_weight_nvfp4_sm120",
    "cake_prepare_minimax_h3_mxfp8_pre_attention",
    "cake_prepare_minimax_h3_nvfp4_pre_attention",
    "cake_prepare_minimax_h3_o_weight_mxfp8",
    "cake_prepare_minimax_h3_o_weight_nvfp4",
    "cake_prepare_minimax_h3_qkv_quantize_pack",
    "cake_prepare_minimax_h3_varlen_attention",
    "cake_prepare_minimax_h3_varlen_nvfp4_attention",
    "cake_quantize_minimax_h3_o_weight_fp8",
    "cake_quantize_minimax_h3_o_weight_nvfp4",
    "cake_quantize_minimax_h3_qkv_weight_fp8",
    "cake_quantize_minimax_h3_qkv_weight_nvfp4",
]
