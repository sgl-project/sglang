"""Cake MiniMax-H3 SM120 (GB202: RTX 5090 / RTX PRO 6000 Blackwell) quantized projections.

FlashInfer entries (``flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_*``,
FlashInfer ``e4f94f9484``; JIT modules
``flashinfer.jit.cake_minimax_h3_sm120_quant_{pre_attention,fc1_swiglu,out_proj}``
built with ``sm120a`` flags only). All compute entries carry
``@supported_compute_capability([120])``; the adapters admit cc 12.0 only.

Pre-attention (``x [M, 5376]`` -> per-kind ``q, k, v [M, 56, *]``, no Ulysses
packing; ``1 <= M <= 2**24``; AdaLN tables ``[rows, 5376]`` with ``rows >= 1``;
any ``eps``):

* ``minimax_h3_fp8_pre_attention`` -- W8A8: ``qkv_weight_q`` E4M3 ``[21504, 5376]``
  + ``qkv_weight_scale`` f32 ``[21504]`` (``quantize_minimax_h3_qkv_weight_fp8``,
  plain row order); per-token activation scale ``RN(amax / 448)``.
* ``minimax_h3_nvfp4_pre_attention`` -- W4A4: ``qkv_weight_q`` uint8 ``[21504, 2688]``,
  ``qkv_weight_sf`` (``21504 * 336`` bytes, swizzled 128x4 direct
  ``fp4_quantize`` output), ``qkv_weight_global_scale`` (float or f32 ``[1]``),
  ``act_global_scale`` f32 ``[1]`` CUDA; ``alpha = 1 / (act_gs * w_gs)`` --
  **pass it explicitly** or FlashInfer derives it with ``.item()`` (host sync).
* ``out_mode``: ``"bf16"`` -> BF16 ``[M, 56, 128]``; ``"e4m3"`` -> E4M3
  ``RN(v / descale)`` with required ``q/k/v_descale``; ``"nvfp4"`` -> uint8
  ``[M, 56, 64]`` + row-major uint8 ``q/k/v_sf [M, 56, 8]`` with required
  ``q/k/v_global_scale``. Descales / global scales given as **tensors are read
  with ``.item()``**: pass Python floats plus every output and the stage-1
  workspaces (``act_q``, ``act_scale`` / ``act_sf``) for CUDA-graph capture;
  otherwise the entry allocates per call and is **not CUDA-graph safe**.
  Returns ``MiniMaxH3PreAttentionOutput(q, k, v, q_sf, k_sf, v_sf)``.

FC1 + SwiGLU (``x [M, 5376]`` -> ``out [M, 14336]``):

* ``minimax_h3_fc1_swiglu_fp8`` -- ``fc1_weight_q`` E4M3 ``[28672, 5376]`` +
  ``fc1_weight_scale`` f32 ``[28672]`` in the **SM120 interleaved row order**
  of ``prepare_minimax_h3_fc1_weight_fp8`` (8 gate rows then 8 up rows per
  column group); optional caller-owned ``workspace_q`` E4M3 ``[M, 5376]`` and
  ``workspace_scale`` f32 ``[M]`` (allocated when omitted).
* ``prepare_minimax_h3_fc1_weight_fp8`` / ``prepare_minimax_h3_fc1_weight_nvfp4_sm120``
  -- offline weight preparation producing the SM120 layouts. **Not
  interchangeable** with the SM100/103 ``prepare_minimax_h3_fc1_weight_mxfp8`` /
  ``..._nvfp4`` combined-tile layouts. The NVFP4 FC1 compute entry on SM120 is
  reached through :func:`sglang.kernels.cake_kernels.diffusion_minimax_h3_proj.minimax_h3_fc1_swiglu_nvfp4`
  (FlashInfer dispatches on cc 12.x).

Out-projection (``attn_out [M, 7168]`` = NHD ``[M, 56, 128]`` viewed row-major,
**not** the Ulysses receive layout of the SM100/103 entry; ``gate [9, 5376]``,
int32 ``gate_index [M]`` (outside ``[0, 9)`` -> ``out = residual``), ``residual``
and ``out [M, 5376]``; ``out = bf16(residual + bf16(gate[idx] * bf16(o)))``):

* ``minimax_h3_fp8_out_proj`` -- ``o_weight_q`` E4M3 ``[5376, 7168]`` +
  ``o_weight_scale`` f32 ``[5376]`` (``quantize_minimax_h3_o_weight_fp8``); one
  launch with producer-warp activation quantization; optional caller-owned
  ``act_q`` E4M3 ``[M, 7168]`` / ``act_scale`` f32 ``[M]``.
* ``minimax_h3_nvfp4_out_proj`` -- ``o_weight_q`` uint8 ``[5376, 3584]``,
  ``o_weight_sf`` (``5376 * 448`` bytes swizzled), ``o_weight_global_scale``,
  ``act_global_scale`` f32 ``[1]`` CUDA; ``alpha`` is computed on the host with
  ``act_global_scale.item()`` **every call** -> not CUDA-graph safe.
* ``quantize_minimax_h3_o_weight_fp8`` / ``quantize_minimax_h3_o_weight_nvfp4``
  -- offline weight preparation (SM120 layouts; not the SM100/103 tiles).

Not supported here: SM90 / SM100 / SM103 (use the SM100/103 adapters), SM121
(sm_120a cubins only), other hidden sizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from sglang.kernels.cake_kernels._support import (
    SM120,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_PRE_ATTENTION_MODULE = (
    "flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention"
)
FI_PRE_ATTENTION_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_sm120_quant_pre_attention"
FI_FC1_MODULE = "flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu"
FI_FC1_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_sm120_quant_fc1_swiglu"
FI_OUT_PROJ_MODULE = "flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj"
FI_OUT_PROJ_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_sm120_quant_out_proj"
ARCHS = (SM120,)

HIDDEN = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QKV_WIDTH = NUM_HEADS * 3 * HEAD_DIM  # 21504
ROPE_DIM = 96
FFN = 14336
FC1_ROWS = 2 * FFN  # 28672
ATTN_DIM = NUM_HEADS * HEAD_DIM  # 7168
ADALN_ROWS = 9
GATE_ROWS = 9
SF_BLOCK = 16
MAX_ROWS = 1 << 24
DEFAULT_EPS = 1.0e-5
OUT_MODES = ("bf16", "e4m3", "nvfp4")

Scalar = Union["torch.Tensor", float]


def _shape(t: torch.Tensor, shape: Tuple[int, ...], dtype) -> bool:
    return tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous()


def _opt(t: Optional[torch.Tensor], shape, dtype, device) -> bool:
    return t is None or (_shape(t, shape, dtype) and t.device == device)


def _scalar_ok(value: Optional[Scalar], device, *, required: bool) -> bool:
    import torch

    if value is None:
        return not required
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return (
        isinstance(value, torch.Tensor)
        and value.numel() == 1
        and value.dtype == torch.float32
        and (value.device == device or not value.is_cuda)
    )


def _eps_ok(eps: float) -> bool:
    return float(eps) > 0.0 and float(eps) == float(eps) and float(eps) != float("inf")


def _norm_inputs_ok(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    *,
    fixed_adaln_rows: Optional[int],
) -> bool:
    import torch

    if not (
        cuda_tensor_on(x, ARCHS)
        and x.ndim == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and _shape(x, (x.shape[0], HIDDEN), torch.bfloat16)
    ):
        return False
    bf16 = torch.bfloat16
    rows = adaln_scale.shape[0] if adaln_scale.ndim == 2 else 0
    if fixed_adaln_rows is not None and rows != fixed_adaln_rows:
        return False
    return (
        rows >= 1
        and _shape(x_norm_weight, (HIDDEN,), bf16)
        and _shape(adaln_scale, (rows, HIDDEN), bf16)
        and _shape(adaln_shift, (rows, HIDDEN), bf16)
        and _shape(adaln_index, (x.shape[0],), torch.int32)
        and all(
            t.device == x.device
            for t in (x_norm_weight, adaln_scale, adaln_shift, adaln_index)
        )
    )


# ---------------------------------------------------------------------------
# Pre-attention (D24 - D27)
# ---------------------------------------------------------------------------


def _pre_attention_outputs_ok(
    m: int,
    device,
    out_mode: str,
    q,
    k,
    v,
    q_sf,
    k_sf,
    v_sf,
    q_descale,
    k_descale,
    v_descale,
    q_global_scale,
    k_global_scale,
    v_global_scale,
) -> bool:
    import torch

    if out_mode not in OUT_MODES:
        return False
    if out_mode == "bf16":
        value_shape, value_dtype = (m, NUM_HEADS, HEAD_DIM), torch.bfloat16
    elif out_mode == "e4m3":
        value_shape, value_dtype = (m, NUM_HEADS, HEAD_DIM), torch.float8_e4m3fn
    else:
        value_shape, value_dtype = (m, NUM_HEADS, HEAD_DIM // 2), torch.uint8
    sf_required = out_mode == "nvfp4"
    sf_shape = (m, NUM_HEADS, HEAD_DIM // SF_BLOCK)
    for t in (q, k, v):
        if not _opt(t, value_shape, value_dtype, device):
            return False
    for sf in (q_sf, k_sf, v_sf):
        if sf is not None and not (
            sf_required and _shape(sf, sf_shape, torch.uint8) and sf.device == device
        ):
            return False
    descales = (q_descale, k_descale, v_descale)
    globals_ = (q_global_scale, k_global_scale, v_global_scale)
    return all(
        _scalar_ok(s, device, required=(out_mode == "e4m3")) for s in descales
    ) and all(_scalar_ok(s, device, required=sf_required) for s in globals_)


def supports_minimax_h3_fp8_pre_attention(
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
    eps: float = DEFAULT_EPS,
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
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        if not (
            flashinfer_module_available(
                FI_PRE_ATTENTION_MODULE, FI_PRE_ATTENTION_JIT_MODULE
            )
            and _norm_inputs_ok(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                fixed_adaln_rows=None,
            )
            and _eps_ok(eps)
        ):
            return False
        m = x.shape[0]
        bf16 = torch.bfloat16
        return (
            _shape(qkv_weight_q, (QKV_WIDTH, HIDDEN), torch.float8_e4m3fn)
            and _shape(qkv_weight_scale, (QKV_WIDTH,), torch.float32)
            and _shape(q_norm_weight, (HEAD_DIM,), bf16)
            and _shape(k_norm_weight, (HEAD_DIM,), bf16)
            and _shape(rope_cos_sin, (m, ROPE_DIM), bf16)
            and all(
                t.device == x.device
                for t in (
                    qkv_weight_q,
                    qkv_weight_scale,
                    q_norm_weight,
                    k_norm_weight,
                    rope_cos_sin,
                )
            )
            and _opt(act_q, (m, HIDDEN), torch.float8_e4m3fn, x.device)
            and _opt(act_scale, (m,), torch.float32, x.device)
            and _pre_attention_outputs_ok(
                m,
                x.device,
                out_mode,
                q,
                k,
                v,
                q_sf,
                k_sf,
                v_sf,
                q_descale,
                k_descale,
                v_descale,
                q_global_scale,
                k_global_scale,
                v_global_scale,
            )
        )
    except Exception:
        return False


def minimax_h3_fp8_pre_attention(
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
    eps: float = DEFAULT_EPS,
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
    """Forward to FlashInfer; returns ``MiniMaxH3PreAttentionOutput``.

    Not CUDA-graph safe unless every output / workspace is supplied and the
    descales / global scales are Python floats.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        minimax_h3_fp8_pre_attention,
    )

    return minimax_h3_fp8_pre_attention(
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


def supports_minimax_h3_nvfp4_pre_attention(
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
    eps: float = DEFAULT_EPS,
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
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        if not (
            flashinfer_module_available(
                FI_PRE_ATTENTION_MODULE, FI_PRE_ATTENTION_JIT_MODULE
            )
            and _norm_inputs_ok(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                fixed_adaln_rows=None,
            )
            and _eps_ok(eps)
        ):
            return False
        m = x.shape[0]
        bf16 = torch.bfloat16
        return (
            _shape(qkv_weight_q, (QKV_WIDTH, HIDDEN // 2), torch.uint8)
            and qkv_weight_sf.dtype == torch.uint8
            and qkv_weight_sf.is_contiguous()
            and qkv_weight_sf.numel() == QKV_WIDTH * (HIDDEN // SF_BLOCK)
            and _scalar_ok(qkv_weight_global_scale, x.device, required=True)
            and isinstance(act_global_scale, torch.Tensor)
            and act_global_scale.is_cuda
            and act_global_scale.device == x.device
            and act_global_scale.numel() == 1
            and act_global_scale.dtype == torch.float32
            and (alpha is None or isinstance(alpha, (int, float)))
            and _shape(q_norm_weight, (HEAD_DIM,), bf16)
            and _shape(k_norm_weight, (HEAD_DIM,), bf16)
            and _shape(rope_cos_sin, (m, ROPE_DIM), bf16)
            and all(
                t.device == x.device
                for t in (
                    qkv_weight_q,
                    qkv_weight_sf,
                    q_norm_weight,
                    k_norm_weight,
                    rope_cos_sin,
                )
            )
            and _opt(act_q, (m, HIDDEN // 2), torch.uint8, x.device)
            and _opt(act_sf, (m, HIDDEN // SF_BLOCK), torch.uint8, x.device)
            and _pre_attention_outputs_ok(
                m,
                x.device,
                out_mode,
                q,
                k,
                v,
                q_sf,
                k_sf,
                v_sf,
                q_descale,
                k_descale,
                v_descale,
                q_global_scale,
                k_global_scale,
                v_global_scale,
            )
        )
    except Exception:
        return False


def minimax_h3_nvfp4_pre_attention(
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
    eps: float = DEFAULT_EPS,
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
    """Forward to FlashInfer; returns ``MiniMaxH3PreAttentionOutput``.

    Pass ``alpha = 1 / (act_global_scale * qkv_weight_global_scale)`` as a float
    to avoid the ``.item()`` host sync; not CUDA-graph safe otherwise.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        minimax_h3_nvfp4_pre_attention,
    )

    return minimax_h3_nvfp4_pre_attention(
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


def supports_quantize_minimax_h3_qkv_weight(qkv_weight: torch.Tensor) -> bool:
    """Admission for both QKV weight quantizers (any CUDA device); never raises."""
    import torch

    try:
        return (
            flashinfer_module_available(FI_PRE_ATTENTION_MODULE)
            and qkv_weight.is_cuda
            and torch.version.cuda is not None
            and qkv_weight.ndim == 2
            and tuple(qkv_weight.shape) == (QKV_WIDTH, HIDDEN)
            and qkv_weight.is_floating_point()
        )
    except Exception:
        return False


def quantize_minimax_h3_qkv_weight_fp8(
    qkv_weight: torch.Tensor, chunk_rows: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(E4M3 [21504, 5376], f32 [21504])`` per-row scales."""
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        quantize_minimax_h3_qkv_weight_fp8,
    )

    return quantize_minimax_h3_qkv_weight_fp8(qkv_weight, chunk_rows=chunk_rows)


def quantize_minimax_h3_qkv_weight_nvfp4(
    qkv_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(uint8 [21504, 2688], swizzled uint8 scales, f32 [1] global scale)``."""
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        quantize_minimax_h3_qkv_weight_nvfp4,
    )

    return quantize_minimax_h3_qkv_weight_nvfp4(qkv_weight)


# ---------------------------------------------------------------------------
# FC1 + SwiGLU (D16 - D18)
# ---------------------------------------------------------------------------


def supports_minimax_h3_fc1_swiglu_fp8(
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
    eps: float = DEFAULT_EPS,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        if not (
            flashinfer_module_available(FI_FC1_MODULE, FI_FC1_JIT_MODULE)
            and _norm_inputs_ok(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                fixed_adaln_rows=ADALN_ROWS,
            )
            and _eps_ok(eps)
        ):
            return False
        m = x.shape[0]
        return (
            _shape(fc1_weight_q, (FC1_ROWS, HIDDEN), torch.float8_e4m3fn)
            and _shape(fc1_weight_scale, (FC1_ROWS,), torch.float32)
            and fc1_weight_q.device == x.device
            and fc1_weight_scale.device == x.device
            and _opt(out, (m, FFN), torch.bfloat16, x.device)
            and _opt(workspace_q, (m, HIDDEN), torch.float8_e4m3fn, x.device)
            and _opt(workspace_scale, (m,), torch.float32, x.device)
        )
    except Exception:
        return False


def minimax_h3_fc1_swiglu_fp8(
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
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [M, 14336]``.

    Supply ``out``, ``workspace_q`` and ``workspace_scale`` for CUDA graphs.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
        minimax_h3_fc1_swiglu_fp8,
    )

    return minimax_h3_fc1_swiglu_fp8(
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


def supports_prepare_minimax_h3_fc1_weight_sm120(fc1_weight: torch.Tensor) -> bool:
    """Admission for both SM120 FC1 weight preparations (any CUDA device); never raises."""
    import torch

    try:
        return (
            flashinfer_module_available(FI_FC1_MODULE)
            and fc1_weight.is_cuda
            and torch.version.cuda is not None
            and _shape(fc1_weight, (FC1_ROWS, HIDDEN), torch.bfloat16)
        )
    except Exception:
        return False


def prepare_minimax_h3_fc1_weight_fp8(
    fc1_weight: torch.Tensor, chunk_rows: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(E4M3 [28672, 5376], f32 [28672])`` in SM120 interleaved row order."""
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
        prepare_minimax_h3_fc1_weight_fp8,
    )

    return prepare_minimax_h3_fc1_weight_fp8(fc1_weight, chunk_rows=chunk_rows)


def prepare_minimax_h3_fc1_weight_nvfp4_sm120(
    fc1_weight: torch.Tensor, w_global_scale: Scalar
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(uint8 [28672, 2688], uint8 [28672 * 336])`` SM120 layout.

    Consumed by the NVFP4 FC1 dispatcher on cc 12.x devices
    (:func:`sglang.kernels.cake_kernels.diffusion_minimax_h3_proj.minimax_h3_fc1_swiglu_nvfp4`).
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
        prepare_minimax_h3_fc1_weight_nvfp4_sm120,
    )

    return prepare_minimax_h3_fc1_weight_nvfp4_sm120(fc1_weight, w_global_scale)


# ---------------------------------------------------------------------------
# Out-projection (D20 - D23)
# ---------------------------------------------------------------------------


def _out_proj_common(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    out: Optional[torch.Tensor],
) -> bool:
    import torch

    if not (
        cuda_tensor_on(attn_out, ARCHS)
        and attn_out.ndim == 2
        and 1 <= attn_out.shape[0] <= MAX_ROWS
        and _shape(attn_out, (attn_out.shape[0], ATTN_DIM), torch.bfloat16)
    ):
        return False
    m = attn_out.shape[0]
    bf16 = torch.bfloat16
    return (
        _shape(gate, (GATE_ROWS, HIDDEN), bf16)
        and _shape(gate_index, (m,), torch.int32)
        and _shape(residual, (m, HIDDEN), bf16)
        and all(t.device == attn_out.device for t in (gate, gate_index, residual))
        and _opt(out, (m, HIDDEN), bf16, attn_out.device)
    )


def supports_minimax_h3_fp8_out_proj(
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
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        m = attn_out.shape[0] if attn_out.ndim == 2 else 0
        return (
            flashinfer_module_available(FI_OUT_PROJ_MODULE, FI_OUT_PROJ_JIT_MODULE)
            and _out_proj_common(attn_out, gate, gate_index, residual, out)
            and _shape(o_weight_q, (HIDDEN, ATTN_DIM), torch.float8_e4m3fn)
            and _shape(o_weight_scale, (HIDDEN,), torch.float32)
            and o_weight_q.device == attn_out.device
            and o_weight_scale.device == attn_out.device
            and _opt(act_q, (m, ATTN_DIM), torch.float8_e4m3fn, attn_out.device)
            and _opt(act_scale, (m,), torch.float32, attn_out.device)
        )
    except Exception:
        return False


def minimax_h3_fp8_out_proj(
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
    """Forward to FlashInfer; returns BF16 ``out [M, 5376]``.

    Supply ``out``, ``act_q`` and ``act_scale`` for CUDA graphs.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        minimax_h3_fp8_out_proj,
    )

    return minimax_h3_fp8_out_proj(
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


def supports_minimax_h3_nvfp4_out_proj(
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
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        m = attn_out.shape[0] if attn_out.ndim == 2 else 0
        return (
            flashinfer_module_available(FI_OUT_PROJ_MODULE, FI_OUT_PROJ_JIT_MODULE)
            and _out_proj_common(attn_out, gate, gate_index, residual, out)
            and _shape(o_weight_q, (HIDDEN, ATTN_DIM // 2), torch.uint8)
            and o_weight_sf.dtype == torch.uint8
            and o_weight_sf.is_contiguous()
            and o_weight_sf.numel() == HIDDEN * (ATTN_DIM // SF_BLOCK)
            and o_weight_q.device == attn_out.device
            and o_weight_sf.device == attn_out.device
            and _scalar_ok(o_weight_global_scale, attn_out.device, required=True)
            and isinstance(act_global_scale, torch.Tensor)
            and act_global_scale.is_cuda
            and act_global_scale.device == attn_out.device
            and act_global_scale.numel() == 1
            and act_global_scale.dtype == torch.float32
            and _opt(act_q, (m, ATTN_DIM // 2), torch.uint8, attn_out.device)
            and _opt(act_sf, (m, ATTN_DIM // SF_BLOCK), torch.uint8, attn_out.device)
        )
    except Exception:
        return False


def minimax_h3_nvfp4_out_proj(
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
    """Forward to FlashInfer; returns BF16 ``out [M, 5376]``.

    FlashInfer derives ``alpha`` on the host with ``act_global_scale.item()``
    every call: this entry is **not CUDA-graph safe**.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        minimax_h3_nvfp4_out_proj,
    )

    return minimax_h3_nvfp4_out_proj(
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


def supports_quantize_minimax_h3_o_weight(o_weight: torch.Tensor) -> bool:
    """Admission for both out-proj weight quantizers (any CUDA device); never raises."""
    import torch

    try:
        return (
            flashinfer_module_available(FI_OUT_PROJ_MODULE)
            and o_weight.is_cuda
            and torch.version.cuda is not None
            and o_weight.ndim == 2
            and tuple(o_weight.shape) == (HIDDEN, ATTN_DIM)
            and o_weight.is_floating_point()
        )
    except Exception:
        return False


def quantize_minimax_h3_o_weight_fp8(
    o_weight: torch.Tensor, chunk_rows: int = 1792
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(E4M3 [5376, 7168], f32 [5376])`` per-output-channel scales."""
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        quantize_minimax_h3_o_weight_fp8,
    )

    return quantize_minimax_h3_o_weight_fp8(o_weight, chunk_rows=chunk_rows)


def quantize_minimax_h3_o_weight_nvfp4(
    o_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(uint8 [5376, 3584], swizzled uint8 scales, f32 [1] global scale)``."""
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        quantize_minimax_h3_o_weight_nvfp4,
    )

    return quantize_minimax_h3_o_weight_nvfp4(o_weight)
