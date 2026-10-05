"""Cake MiniMax-H3 pre-attention projections (SM100a / SM103a) via FlashInfer.

FlashInfer entries (``flashinfer.diffusion_ops``; the BF16 entry at FlashInfer
``bd94c5806``, the quantized chains at ``e4f94f9484``):

* ``minimax_h3_bf16_pre_attention`` -- input RMSNorm + indexed AdaLN + BF16 QKV
  projection + per-head Q/K RMSNorm + partial 3-D split-half NeoX RoPE +
  destination-major pack ``out [P, M, 56 // P, 3, 128]`` as two launches (the
  normalized, modulated activation goes through a BF16 ``[M, 5376]`` workspace
  that the persistent QKV GEMM streams through TMA; its epilogue applies the
  Q/K norms, RoPE and pack). JIT module
  ``flashinfer.jit.cake_minimax_h3_bf16_pre_attention``. Caller-owned ``out``;
  the workspace is a per-call allocation from the caching allocator (FlashInfer
  default), freed when the stage returns, so the route holds no persistent
  scratch between calls; no host sync, CUDA-graph capturable.
  Takes the diffusion engine's own operands: AdaLN tables ``[rows, 5376]`` with
  any ``rows >= 1`` and a 16-byte-aligned row pitch (column chunks of the
  ``[rows, 6 * 5376]`` modulation projection pass as they are), int64
  ``adaln_index [M]``, RoPE as the shared ``(rope_cos_sin [S, 96],
  rope_positions int64 [M])`` pair (``rope_positions=None`` = identity, then
  ``S >= M``), separate ``eps`` (input norm) and ``qk_eps`` (Q/K norms, default
  ``eps``); no host copies.
* ``prepare_minimax_h3_mxfp8_pre_attention`` -> ``PreparedMiniMaxH3Mxfp8PreAttention``
  -- 3-launch chain (norm/AdaLN/MXFP8 quant -> CUTLASS MXFP8 GEMM with a tactic
  pinned at prepare -> QK-norm/RoPE/MXFP8 pack). One JIT loader for both
  targets, ``flashinfer.jit.cake_minimax_h3_mxfp8`` (FlashInfer ``e6c0d39f6``
  replaced the ``..._mxfp8_pre_attention{,_sm100a,_sm103a}`` loaders); the
  stages are compiled per exact ``(M, P)`` from an **exact** route table
  (44 pairs, see :data:`MXFP8_ROUTES`); any other ``M`` raises in FlashInfer.
  ``__call__()`` zeroes ``out_sf`` / ``activation_sf`` on device and performs
  no allocation: graph-capturable.
* ``prepare_minimax_h3_nvfp4_pre_attention`` -> ``PreparedMiniMaxH3Nvfp4PreAttention``
  -- 2-launch chain (norm/AdaLN/NVFP4 quant -> fused tcgen05 NVFP4 QKV GEMM
  with QK-norm/RoPE/NVFP4 pack epilogue). One JIT loader for both targets,
  ``flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention`` (FlashInfer
  ``1fc76f71b`` removed the ``_sm100a`` / ``_sm103a`` split loaders). Routed
  by ``P`` only; ``M`` is a runtime parameter. ``alpha = 1 / (x_gs * w_gs)``
  and the CTA-pair repack of ``qkv_weight_sf`` are derived once at prepare
  (new scales => new prepare).
* ``prepare_minimax_h3_qkv_quantize_pack`` -> ``PreparedMiniMaxH3QkvQuantizePack``
  and the one-shot ``minimax_h3_qkv_quantize_pack`` -- BF16 ``q, k, v
  [M, 56, 128]`` (or kind slices of one ``[M, 56, 3, 128]`` projection) ->
  Ulysses send buffer ``(out_q, out_sf)``, byte-identical to the MXFP8 / NVFP4
  pre-attention output. One JIT loader for both targets,
  ``flashinfer.jit.cake_minimax_h3_qkv_quantize_pack`` (one program per
  format, ``P`` a compile-line define). The prepared runner is one launch, no
  allocation, no host clear. The one-shot form re-prepares and may allocate:
  **not CUDA-graph safe**.

The generated MXFP8 / NVFP4 stages take their TMA tensor maps by value: there
is no descriptor workspace. FlashInfer keeps the ``norm_descriptor_workspace``
/ ``post_descriptor_workspace`` / ``gemm_descriptor_workspace`` keywords and
raises ``ValueError`` for any non-``None`` value; this adapter does not expose
them.

Shared contract: BF16 ``x [M, 5376]``, ``x_norm_weight [5376]``, BF16
``q/k_norm_weight [128]``, RoPE cols ``[0, 48)`` cos, ``[48, 96)`` sin (head
dims ``[96, 128)`` pass through), ``P`` (``ulysses_degree``) in ``{1, 2, 4, 8}``.
The quantized (MXFP8 / NVFP4 / quantize-and-pack) chains keep the exact-shape
contract: contiguous BF16 AdaLN tables ``[9, 5376]``, int32 ``adaln_index [M]``
(out-of-range -> zero row, device-guarded), BF16 per-row ``rope_cos_sin
[M, 96]``, ``eps == 1e-5``. Every output / workspace is caller-owned. Prepared
runners are bound to the tensor objects given at prepare (contents may change
between launches).

Built for exact compute capability 10.0 / 10.3 only (sm_100a / sm_103a).

Not supported here (keep the existing SGLang path): any other hidden size or
head configuration, SM90, SM120 (use the separate SM120 entries in
:mod:`sglang.kernels.cake_kernels.diffusion_minimax_h3_sm120`), MXFP8 ``(M, P)``
pairs outside the route table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Tuple

from sglang.kernels.cake_kernels._support import (
    SM100,
    SM103,
    cuda_tensor_on,
    flashinfer_module_available,
    table_view_ok,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.diffusion_ops.minimax_h3"
FI_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_bf16_pre_attention"
FI_MXFP8_MODULE = "flashinfer.diffusion_ops.cake_minimax_h3_mxfp8"
FI_MXFP8_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_mxfp8"
FI_NVFP4_MODULE = "flashinfer.diffusion_ops.cake_minimax_h3_nvfp4"
FI_NVFP4_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention"
FI_QKV_PACK_MODULE = "flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack"
FI_QKV_PACK_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_qkv_quantize_pack"
ARCHS = (SM100, SM103)

HIDDEN = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QKV_KINDS = 3
QKV_WIDTH = NUM_HEADS * QKV_KINDS * HEAD_DIM  # 21504
ROPE_DIM = 96
# Exact table row count of the quantized (MXFP8 / NVFP4) chains; the BF16 entry
# takes any ``rows >= 1``.
ADALN_ROWS = 9
# Input-norm eps the quantized chains are compiled for; the BF16 entry takes
# ``eps`` / ``qk_eps`` as runtime floats.
EPS = 1.0e-5
# BF16 tables / rope cache: row pitch in elements that keeps rows 16-byte aligned.
TABLE_ALIGN_ELEMENTS = 8
ULYSSES_DEGREES = (1, 2, 4, 8)
PACK_FORMATS = ("nvfp4", "mxfp8")

# Exact (M, P) routes of the MXFP8 pre-attention chain, shared by sm_100a and
# sm_103a at FlashInfer e4f94f9484 (44 pairs). Mirrors
# ``flashinfer.jit.cake_minimax_h3_mxfp8.MINIMAX_H3_MXFP8_SHAPES``.
MXFP8_ROUTES: FrozenSet[Tuple[int, int]] = frozenset(
    [(m, 8) for m in (1, 127, 128, 129, 4184, 4816, 4823, 4824, 4825, 4832)]
    + [(m, 8) for m in (6096, 7368, 9280, 13744)]
    + [(m, 4) for m in (8368, 9632, 9647, 9648, 9649, 9664, 12192, 14736)]
    + [(m, 4) for m in (18560, 27488)]
    + [(m, 2) for m in (16736, 19264, 19295, 19296, 19297, 19328, 24384)]
    + [(m, 2) for m in (29472, 37120, 54976)]
    + [(m, 1) for m in (33472, 38528, 38591, 38592, 38593, 38656, 48768)]
    + [(m, 1) for m in (58944, 74240, 109952)]
)


def _round_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _shape(t: torch.Tensor, shape: Tuple[int, ...], dtype) -> bool:
    return tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous()


def _common_pre_attention_inputs(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    P: int,
    eps: float,
) -> bool:
    import torch

    if not (
        cuda_tensor_on(x, ARCHS)
        and x.ndim == 2
        and x.shape[0] > 0
        and _shape(x, (x.shape[0], HIDDEN), torch.bfloat16)
    ):
        return False
    m = x.shape[0]
    bf16 = torch.bfloat16
    return (
        not isinstance(P, bool)
        and P in ULYSSES_DEGREES
        and float(eps) == EPS
        and _shape(x_norm_weight, (HIDDEN,), bf16)
        and _shape(adaln_scale, (ADALN_ROWS, HIDDEN), bf16)
        and _shape(adaln_shift, (ADALN_ROWS, HIDDEN), bf16)
        and _shape(adaln_index, (m,), torch.int32)
        and _shape(q_norm_weight, (HEAD_DIM,), bf16)
        and _shape(k_norm_weight, (HEAD_DIM,), bf16)
        and _shape(rope_cos_sin, (m, ROPE_DIM), bf16)
        and all(
            t.device == x.device
            for t in (
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                q_norm_weight,
                k_norm_weight,
                rope_cos_sin,
            )
        )
    )


# ---------------------------------------------------------------------------
# BF16 pre-attention (D01)
# ---------------------------------------------------------------------------


def _bf16_pre_attention_inputs(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    rope_positions: Optional[torch.Tensor],
    *,
    P: int,
    eps: float,
    qk_eps: Optional[float],
) -> bool:
    """The engine-operand contract of the BF16 entry (FlashInfer ``bd94c5806``)."""
    import torch

    if not (
        cuda_tensor_on(x, ARCHS)
        and x.ndim == 2
        and x.shape[0] > 0
        and _shape(x, (x.shape[0], HIDDEN), torch.bfloat16)
    ):
        return False
    m = int(x.shape[0])
    bf16 = torch.bfloat16
    device = x.device
    float(eps)
    if qk_eps is not None:
        float(qk_eps)

    def rope_ok() -> bool:
        # The cache is gathered on the device: identity positions need S >= M,
        # explicit positions are an int64 [M] row map into the cache.
        if rope_positions is None:
            return int(rope_cos_sin.shape[0]) >= m
        return (
            _shape(rope_positions, (m,), torch.int64)
            and rope_positions.device == device
        )

    return (
        not isinstance(P, bool)
        and P in ULYSSES_DEGREES
        and _shape(x_norm_weight, (HIDDEN,), bf16)
        and x_norm_weight.device == device
        and table_view_ok(
            adaln_scale,
            rows_min=1,
            cols=HIDDEN,
            dtype=bf16,
            device=device,
            align_elements=TABLE_ALIGN_ELEMENTS,
        )
        and table_view_ok(
            adaln_shift,
            rows_min=1,
            cols=HIDDEN,
            dtype=bf16,
            device=device,
            align_elements=TABLE_ALIGN_ELEMENTS,
        )
        and adaln_scale.shape[0] == adaln_shift.shape[0]
        and _shape(adaln_index, (m,), torch.int64)
        and adaln_index.device == device
        and _shape(q_norm_weight, (HEAD_DIM,), bf16)
        and _shape(k_norm_weight, (HEAD_DIM,), bf16)
        and q_norm_weight.device == device
        and k_norm_weight.device == device
        and table_view_ok(
            rope_cos_sin,
            rows_min=1,
            cols=ROPE_DIM,
            dtype=bf16,
            device=device,
            align_elements=TABLE_ALIGN_ELEMENTS,
        )
        and rope_cos_sin.is_contiguous()
        and rope_ok()
    )


def supports_minimax_h3_bf16_pre_attention(
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
    eps: float = EPS,
    qk_eps: Optional[float] = None,
    rope_positions: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    ``adaln_scale`` / ``adaln_shift`` are any ``[rows >= 1, 5376]`` BF16 views
    with a 16-byte-aligned row pitch, ``adaln_index`` is int64, ``rope_cos_sin``
    is the shared ``[S, 96]`` cache indexed by int64 ``rope_positions [M]``
    (``None``: identity, ``S >= M``); ``eps`` / ``qk_eps`` are free floats.
    """
    import torch

    try:
        return (
            flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
            and _bf16_pre_attention_inputs(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                q_norm_weight,
                k_norm_weight,
                rope_cos_sin,
                rope_positions,
                P=ulysses_degree,
                eps=eps,
                qk_eps=qk_eps,
            )
            and _shape(qkv_weight, (QKV_WIDTH, HIDDEN), torch.bfloat16)
            and qkv_weight.device == x.device
            and _shape(
                out,
                (
                    ulysses_degree,
                    x.shape[0],
                    NUM_HEADS // ulysses_degree,
                    QKV_KINDS,
                    HEAD_DIM,
                ),
                torch.bfloat16,
            )
            and out.device == x.device
        )
    except Exception:
        return False


def minimax_h3_bf16_pre_attention(
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
    eps: float = EPS,
    qk_eps: Optional[float] = None,
    rope_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to FlashInfer; returns the caller-owned ``out``.

    The tables, the index and the ``(rope_cos_sin, rope_positions)`` pair are
    handed to the kernels as they are (no host copies).  The two-launch stage
    is faster than the segmented norm + cuBLAS + fused-postprocess chain at
    every production center for ``ulysses_degree in {1, 2, 4, 8}`` (SM100a and
    SM103a), so no segmented path is kept for ``P=1``.  The BF16 ``[M, 5376]``
    activation workspace is a per-call allocation inside FlashInfer (freed on
    return): the route keeps no persistent scratch, so the pipeline's peak
    memory is not raised by a cached buffer.
    """
    from flashinfer.diffusion_ops.minimax_h3 import minimax_h3_bf16_pre_attention

    return minimax_h3_bf16_pre_attention(
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
        qk_eps=qk_eps,
        rope_positions=rope_positions,
    )


# ---------------------------------------------------------------------------
# MXFP8 prepared pre-attention (D04 / D05)
# ---------------------------------------------------------------------------


def mxfp8_out_sf_numel(M: int, P: int) -> int:
    """Per-destination MXFP8 scale stride: ``round_up(M * (56 / P) * 3, 128) * 4``."""
    return _round_up(M * (NUM_HEADS // P) * QKV_KINDS, 128) * (HEAD_DIM // 32)


def mxfp8_activation_sf_numel(M: int) -> int:
    """MXFP8 activation scale buffer: ``round_up(M, 128) * 168`` bytes."""
    return _round_up(M, 128) * (HIDDEN // 32)


def supports_prepare_minimax_h3_mxfp8_pre_attention(
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
    eps: float = EPS,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    Includes the exact ``(M, P)`` route table and the FlashInfer
    ``gemm_base.DEFAULT_WORKSPACE_SIZE`` lower bound on ``gemm_workspace``
    (read lazily; ``False`` when FlashInfer is absent).
    """
    import torch

    try:
        if not (
            flashinfer_module_available(FI_MXFP8_MODULE, FI_MXFP8_JIT_MODULE)
            and _common_pre_attention_inputs(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                q_norm_weight,
                k_norm_weight,
                rope_cos_sin,
                P=P,
                eps=eps,
            )
        ):
            return False
        m = x.shape[0]
        if (m, P) not in MXFP8_ROUTES:
            return False
        from flashinfer.gemm import gemm_base

        e4m3 = torch.float8_e4m3fn
        return (
            _shape(qkv_weight_q, (QKV_WIDTH, HIDDEN), e4m3)
            and _shape(qkv_weight_sf, (QKV_WIDTH * (HIDDEN // 32),), torch.uint8)
            and _shape(out_q, (P, m, NUM_HEADS // P, QKV_KINDS, HEAD_DIM), e4m3)
            and _shape(out_sf, (P, mxfp8_out_sf_numel(m, P)), torch.uint8)
            and _shape(activation_q, (m, HIDDEN), e4m3)
            and _shape(activation_sf, (mxfp8_activation_sf_numel(m),), torch.uint8)
            and _shape(qkv_bf16, (m, QKV_WIDTH), torch.bfloat16)
            and gemm_workspace.dtype == torch.uint8
            and gemm_workspace.numel() >= int(gemm_base.DEFAULT_WORKSPACE_SIZE)
            and all(
                t.device == x.device
                for t in (
                    qkv_weight_q,
                    qkv_weight_sf,
                    out_q,
                    out_sf,
                    activation_q,
                    activation_sf,
                    qkv_bf16,
                    gemm_workspace,
                )
            )
        )
    except Exception:
        return False


def prepare_minimax_h3_mxfp8_pre_attention(
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
    eps: float = EPS,
) -> Any:
    """Forward to FlashInfer; returns ``PreparedMiniMaxH3Mxfp8PreAttention``.

    Call the returned object (``runner() -> (out_q, out_sf)``) to launch the
    prepared chain. Prepare runs the norm stage once and autotunes the CUTLASS
    MXFP8 GEMM tactic; the runner is allocation-free afterwards. The stages
    take their tensor maps by value (no descriptor workspace).
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_mxfp8 import (
        prepare_minimax_h3_mxfp8_pre_attention,
    )

    return prepare_minimax_h3_mxfp8_pre_attention(
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


# ---------------------------------------------------------------------------
# NVFP4 prepared pre-attention (D06 / D07)
# ---------------------------------------------------------------------------


def nvfp4_out_sf_numel(M: int, P: int) -> int:
    """Per-destination NVFP4 scale stride: ``round_up(M * (56 / P) * 3, 128) * 8``."""
    return _round_up(M * (NUM_HEADS // P) * QKV_KINDS, 128) * (HEAD_DIM // 16)


def nvfp4_activation_sf_numel(M: int) -> int:
    """NVFP4 activation scale buffer: ``round_up(M, 128) * 336`` bytes."""
    return _round_up(M, 128) * (HIDDEN // 16)


def supports_prepare_minimax_h3_nvfp4_pre_attention(
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
    eps: float = EPS,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    Both stages take their TMA tensor maps by value: no descriptor workspace
    is required (or accepted).
    """
    import torch

    try:
        if not (
            flashinfer_module_available(FI_NVFP4_MODULE, FI_NVFP4_JIT_MODULE)
            and _common_pre_attention_inputs(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                q_norm_weight,
                k_norm_weight,
                rope_cos_sin,
                P=P,
                eps=eps,
            )
        ):
            return False
        m = x.shape[0]
        u8 = torch.uint8
        f32 = torch.float32
        return (
            _shape(x_global_scale, (1,), f32)
            and _shape(w_global_scale, (1,), f32)
            and _shape(out_global_scale, (1,), f32)
            and _shape(qkv_weight_q, (QKV_WIDTH, HIDDEN // 2), u8)
            and _shape(qkv_weight_sf, (QKV_WIDTH * (HIDDEN // 16),), u8)
            and _shape(out_q, (P, m, NUM_HEADS // P, QKV_KINDS, HEAD_DIM // 2), u8)
            and _shape(out_sf, (P, nvfp4_out_sf_numel(m, P)), u8)
            and _shape(activation_q, (m, HIDDEN // 2), u8)
            and _shape(activation_sf, (nvfp4_activation_sf_numel(m),), u8)
            and all(
                t.device == x.device
                for t in (
                    x_global_scale,
                    w_global_scale,
                    out_global_scale,
                    qkv_weight_q,
                    qkv_weight_sf,
                    out_q,
                    out_sf,
                    activation_q,
                    activation_sf,
                )
            )
        )
    except Exception:
        return False


def prepare_minimax_h3_nvfp4_pre_attention(
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
    eps: float = EPS,
) -> Any:
    """Forward to FlashInfer; returns ``PreparedMiniMaxH3Nvfp4PreAttention``.

    ``runner() -> (out_q, out_sf)``. Optional debug outputs (post-RoPE BF16 Q
    and K ``[M, 56, 128]`` and the stage-1 AdaLN intermediate ``[M, 5376]``)
    must be supplied all together or not at all. The stages take their TMA
    tensor maps by value (no descriptor workspace).
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_nvfp4 import (
        prepare_minimax_h3_nvfp4_pre_attention,
    )

    return prepare_minimax_h3_nvfp4_pre_attention(
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


# ---------------------------------------------------------------------------
# QKV quantize-and-pack (D09 / D10 / D11 / D12)
# ---------------------------------------------------------------------------


def minimax_h3_qkv_pack_output_shapes(
    M: int, P: int, format: str
) -> Dict[str, Tuple[Tuple[int, ...], Any]]:
    """``{"out_q": (shape, dtype), "out_sf": (shape, dtype)}`` for ``(M, P, format)``.

    Host helper (FlashInfer ``minimax_h3_qkv_pack_output_shapes``) for
    allocating the caller-owned send buffers before ``prepare``.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack import (
        minimax_h3_qkv_pack_output_shapes,
    )

    return minimax_h3_qkv_pack_output_shapes(M, P, format)


def _qkv_sources_ok(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    import torch

    if not cuda_tensor_on(q, ARCHS):
        return False
    m = q.shape[0] if q.ndim == 3 else -1
    for t in (q, k, v):
        if not (
            t.dtype == torch.bfloat16
            and t.ndim == 3
            and tuple(t.shape) == (m, NUM_HEADS, HEAD_DIM)
            and t.device == q.device
            and t.stride(2) == 1
            and (t.stride(1), t.stride(0)) == (q.stride(1), q.stride(0))
            and t.stride(1) >= HEAD_DIM
            and t.stride(1) % 16 == 0
            and t.stride(0) % 16 == 0
            and t.stride(0) >= NUM_HEADS * t.stride(1)
            and t.stride(0) < 2**31
            and t.data_ptr() % 32 == 0
        ):
            return False
    return m > 0


def _qkv_pack_outputs_ok(
    m: int,
    P: int,
    format: str,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    out_global_scale: Optional[torch.Tensor],
    device,
) -> bool:
    import torch

    if format == "nvfp4":
        values_ok = _shape(
            out_q, (P, m, NUM_HEADS // P, QKV_KINDS, HEAD_DIM // 2), torch.uint8
        )
        sf_ok = _shape(out_sf, (P, nvfp4_out_sf_numel(m, P)), torch.uint8)
        gs_ok = (
            out_global_scale is not None
            and _shape(out_global_scale, (1,), torch.float32)
            and out_global_scale.device == device
        )
    else:
        values_ok = _shape(
            out_q, (P, m, NUM_HEADS // P, QKV_KINDS, HEAD_DIM), torch.float8_e4m3fn
        )
        sf_ok = _shape(out_sf, (P, mxfp8_out_sf_numel(m, P)), torch.uint8)
        gs_ok = out_global_scale is None
    return (
        values_ok
        and sf_ok
        and gs_ok
        and out_q.device == device
        and out_sf.device == device
    )


def supports_prepare_minimax_h3_qkv_quantize_pack(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    P: int,
    format: str,
    out_global_scale: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    try:
        return (
            flashinfer_module_available(FI_QKV_PACK_MODULE, FI_QKV_PACK_JIT_MODULE)
            and not isinstance(P, bool)
            and P in ULYSSES_DEGREES
            and format in PACK_FORMATS
            and _qkv_sources_ok(q, k, v)
            and _qkv_pack_outputs_ok(
                q.shape[0], P, format, out_q, out_sf, out_global_scale, q.device
            )
        )
    except Exception:
        return False


def prepare_minimax_h3_qkv_quantize_pack(
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
    """Forward to FlashInfer; returns ``PreparedMiniMaxH3QkvQuantizePack``.

    ``runner() -> (out_q, out_sf)`` is exactly one kernel launch on the torch
    stream with no allocation and no host clear (CUDA-graph capturable).
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack import (
        prepare_minimax_h3_qkv_quantize_pack,
    )

    return prepare_minimax_h3_qkv_quantize_pack(
        q=q,
        k=k,
        v=v,
        out_q=out_q,
        out_sf=out_sf,
        P=P,
        format=format,
        out_global_scale=out_global_scale,
    )


def supports_minimax_h3_qkv_quantize_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    P: int,
    format: str,
    out_global_scale: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check for the one-shot form; never raises.

    Outputs are optional (allocated by FlashInfer when omitted); when both are
    given they must match the send-buffer ABI.
    """
    import torch

    try:
        if not (
            flashinfer_module_available(FI_QKV_PACK_MODULE, FI_QKV_PACK_JIT_MODULE)
            and not isinstance(P, bool)
            and P in ULYSSES_DEGREES
            and format in PACK_FORMATS
            and _qkv_sources_ok(q, k, v)
        ):
            return False
        if format == "nvfp4":
            if not (
                out_global_scale is not None
                and _shape(out_global_scale, (1,), torch.float32)
                and out_global_scale.device == q.device
            ):
                return False
        elif out_global_scale is not None:
            return False
        if out_q is None and out_sf is None:
            return True
        if out_q is None or out_sf is None:
            return False
        return _qkv_pack_outputs_ok(
            q.shape[0], P, format, out_q, out_sf, out_global_scale, q.device
        )
    except Exception:
        return False


def minimax_h3_qkv_quantize_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    P: int,
    format: str,
    out_global_scale: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(out_q, out_sf)``.

    Re-prepares on every call and allocates the outputs when omitted: not
    CUDA-graph safe. Use :func:`prepare_minimax_h3_qkv_quantize_pack` for
    graphs.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack import (
        minimax_h3_qkv_quantize_pack,
    )

    return minimax_h3_qkv_quantize_pack(
        q,
        k,
        v,
        P,
        format,
        out_global_scale=out_global_scale,
        out_q=out_q,
        out_sf=out_sf,
    )
