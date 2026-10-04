"""Cake MiniMax-H3 FC1+SwiGLU and gated-residual out-projection (SM100a / SM103a).

FlashInfer entries (``flashinfer.diffusion_ops.minimax_h3_fc1_swiglu`` and
``flashinfer.diffusion_ops.minimax_h3_out_proj``, FlashInfer ``f62ffa92a12``;
JIT loaders ``flashinfer.jit.minimax_h3_fc1_swiglu`` /
``flashinfer.jit.minimax_h3_out_proj`` compile the ``cake_minimax_h3_*_sm100a /
_sm103a.cu`` sources).

Both families take the diffusion engine's own table operands: BF16 AdaLN /
gate tables ``[rows, 5376]`` with any ``rows >= 1`` and a 16-byte-aligned row
pitch (column chunks of the ``[rows, 6 * 5376]`` modulation projection pass as
they are), int64 row indices ``[M]`` (an index outside ``[0, rows)`` is guarded
on the device: zero modulated row for FC1, ``out = residual`` for the
out-projection) and a free ``eps``; nothing is copied on the host.

FC1 + SwiGLU (``x [M, 5376]`` -> ``out [M, 14336]``, ``y = bf16(silu(gate) * up)``
with ``fc1_weight`` gate rows ``[0, 14336)`` first, then up rows):

* ``minimax_h3_fc1_swiglu`` -- BF16 weight; one-warp-per-row norm/AdaLN into a
  BF16 ``workspace [M, 5376]`` then a persistent 2-CTA tcgen05 GEMM with the
  SwiGLU epilogue. Pass ``workspace`` and ``out`` for CUDA-graph capture.
* ``minimax_h3_fc1_swiglu_mxfp8`` -- ``fc1_weight_q`` E4M3 ``[28672, 5376]`` +
  ``fc1_scale_tiles`` (4,816,896 bytes, combined 256-row gate/up tiles from
  ``prepare_minimax_h3_fc1_weight_mxfp8``); activation quantization bit-exact
  with ``flashinfer.mxfp8_quantize``. ``workspace_q`` E4M3 ``[M, 5376]`` and
  ``workspace_sf`` (``>= mxfp8_fc1_activation_scale_workspace_bytes(M)``) are
  **allocated zeroed per call when omitted** -- supply them for graphs.
* ``minimax_h3_fc1_swiglu_nvfp4`` -- arch dispatcher: cc 10.0/10.3 run the
  tcgen05 route (``fc1_weight_q`` uint8 ``[28672, 2688]``, ``fc1_scale_tiles``
  11,010,048 bytes from ``prepare_minimax_h3_fc1_weight_nvfp4``, ``a_global_scale``
  and ``alpha`` as f32 ``[1]``); cc 12.x delegates to the SM120 ``mma.sync``
  route whose prepared weight layout differs (prepare on the arch that runs).
* ``prepare_minimax_h3_fc1_weight_mxfp8`` / ``prepare_minimax_h3_fc1_weight_nvfp4``
  -- offline weight preparation (pure torch + FlashInfer quantizers). The
  outputs are **arch-specific**: SM100/103 combined tiles are not
  interchangeable with the SM120 layouts of ``prepare_minimax_h3_fc1_weight_fp8``
  / ``prepare_minimax_h3_fc1_weight_nvfp4_sm120``.

Out-projection (``attn_out [P, M, 56 // P, 128]`` Ulysses receive layout,
``o_weight [5376, 7168]``, ``gate [rows, 5376]`` with ``5376 <= stride(0) <
2**32``, int64 ``gate_index [M]`` (outside ``[0, rows)`` -> gate 0 ->
``out = residual``), ``residual [M, 5376]`` ->
``out = bf16(residual + bf16(gate * bf16(A @ W^T)))``):

* ``minimax_h3_out_proj`` -- BF16, one persistent 2-CTA tcgen05 launch; only
  ``out`` may be allocated, so supply it for graphs.
* ``minimax_h3_out_proj_mxfp8`` / ``minimax_h3_out_proj_nvfp4`` -- two launches
  (one-warp-per-row activation quant reading the receive layout -> GEMM).
  ``workspace_q`` / ``workspace_sf`` (``>= *_out_proj_activation_scale_workspace_bytes(M)``)
  are allocated zeroed per call when omitted. The NVFP4 form takes
  ``a_global_scale`` and ``alpha`` as f32 ``[1]`` device tensors.
* ``prepare_minimax_h3_o_weight_mxfp8`` / ``prepare_minimax_h3_o_weight_nvfp4``
  -- offline weight preparation for the SM100/103 routes (not the layout the
  SM120 ``quantize_minimax_h3_o_weight_*`` entries expect).

All compute entries require exact compute capability 10.0 / 10.3 (FlashInfer
raises ``RuntimeError`` otherwise, except the NVFP4 FC1 dispatcher on 12.x),
CUDA >= 12.9 and ``1 <= M <= 2**24``.

Not supported here: other hidden sizes, SM90, SM120 (use
:mod:`sglang.kernels.cake_kernels.diffusion_minimax_h3_sm120`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from sglang.kernels.cake_kernels._support import (
    SM100,
    SM103,
    SM120,
    cuda_tensor_on,
    flashinfer_module_available,
    table_view_ok,
)

if TYPE_CHECKING:
    import torch

FI_FC1_MODULE = "flashinfer.diffusion_ops.minimax_h3_fc1_swiglu"
FI_FC1_JIT_MODULE = "flashinfer.jit.minimax_h3_fc1_swiglu"
FI_OUT_PROJ_MODULE = "flashinfer.diffusion_ops.minimax_h3_out_proj"
FI_OUT_PROJ_JIT_MODULE = "flashinfer.jit.minimax_h3_out_proj"
FI_SM120_FC1_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_sm120_quant_fc1_swiglu"
ARCHS = (SM100, SM103)
# The cc 12.x branch of the NVFP4 FC1 dispatcher loads an sm_120a-only cubin.
NVFP4_FC1_DISPATCH_ARCHS = (SM100, SM103, SM120)

HIDDEN = 5376
FFN = 14336
FC1_ROWS = 2 * FFN  # 28672
NUM_HEADS = 56
HEAD_DIM = 128
ATTN_DIM = NUM_HEADS * HEAD_DIM  # 7168
EPS = 1.0e-5  # default only; any float is admitted
TABLE_ALIGN_ELEMENTS = 8  # 16-byte row pitch / data pointer alignment (BF16)
MAX_ROWS = 1 << 24
SEQUENCE_PARALLEL_DEGREES = (1, 2, 4, 8)
MXFP8_FC1_SCALE_TILE_BYTES = 112 * 42 * 1024  # 4_816_896
NVFP4_FC1_SCALE_TILE_BYTES = 128 * 84 * 1024  # 11_010_048
MXFP8_O_SCALE_TILE_BYTES = 21 * 56 * 1024  # 1_204_224
NVFP4_O_SCALE_TILE_BYTES = 21 * 112 * 1024  # 2_408_448

Scalar = Union["torch.Tensor", float]


def _m_tiles(rows: int) -> int:
    tiles = -(-rows // 128)
    return tiles + tiles % 2


def mxfp8_fc1_activation_scale_workspace_bytes(rows: int) -> int:
    """``workspace_sf`` bytes of the MXFP8 FC1 route: ``m_tiles(M) * 42 * 512``."""
    return _m_tiles(int(rows)) * (HIDDEN // 32 // 4) * 512


def nvfp4_fc1_activation_scale_workspace_bytes(rows: int) -> int:
    """``workspace_sf`` bytes of the NVFP4 FC1 route: ``m_tiles(M) * 84 * 512``."""
    return _m_tiles(int(rows)) * (HIDDEN // 16 // 4) * 512


def mxfp8_out_proj_activation_scale_workspace_bytes(rows: int) -> int:
    """``workspace_sf`` bytes of the MXFP8 out-proj route: ``m_tiles(M) * 56 * 512``."""
    return _m_tiles(int(rows)) * (ATTN_DIM // 32 // 4) * 512


def nvfp4_out_proj_activation_scale_workspace_bytes(rows: int) -> int:
    """``workspace_sf`` bytes of the NVFP4 out-proj route: ``m_tiles(M) * 112 * 512``."""
    return _m_tiles(int(rows)) * (ATTN_DIM // 16 // 4) * 512


def _shape(t: torch.Tensor, shape: Tuple[int, ...], dtype) -> bool:
    return tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous()


def _opt(t: Optional[torch.Tensor], shape, dtype, device) -> bool:
    return t is None or (_shape(t, shape, dtype) and t.device == device)


def _opt_min_bytes(t: Optional[torch.Tensor], nbytes: int, device) -> bool:
    import torch

    return t is None or (
        t.dtype == torch.uint8
        and t.is_contiguous()
        and t.numel() >= nbytes
        and t.device == device
    )


def _f32_scalar(t: Scalar, device) -> bool:
    import torch

    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return True
    return (
        isinstance(t, torch.Tensor)
        and t.numel() == 1
        and t.dtype == torch.float32
        and (t.device == device or not t.is_cuda)
    )


def _table(t: torch.Tensor, device) -> bool:
    """``[rows >= 1, 5376]`` BF16 table view with a 16-byte-aligned row pitch."""
    import torch

    return table_view_ok(
        t,
        rows_min=1,
        cols=HIDDEN,
        dtype=torch.bfloat16,
        device=device,
        align_elements=TABLE_ALIGN_ELEMENTS,
    )


def _fc1_common(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    out: Optional[torch.Tensor],
    eps: float,
    archs,
) -> bool:
    import torch

    if not (
        cuda_tensor_on(x, archs)
        and x.ndim == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and _shape(x, (x.shape[0], HIDDEN), torch.bfloat16)
    ):
        return False
    m = x.shape[0]
    bf16 = torch.bfloat16
    float(eps)
    return (
        _shape(x_norm_weight, (HIDDEN,), bf16)
        and _table(adaln_scale, x.device)
        and _table(adaln_shift, x.device)
        and adaln_scale.shape[0] == adaln_shift.shape[0]
        and _shape(adaln_index, (m,), torch.int64)
        and x_norm_weight.device == x.device
        and adaln_index.device == x.device
        and _opt(out, (m, FFN), bf16, x.device)
    )


# ---------------------------------------------------------------------------
# FC1 + SwiGLU (D28 - D33)
# ---------------------------------------------------------------------------


def supports_minimax_h3_fc1_swiglu(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        return (
            flashinfer_module_available(FI_FC1_MODULE, FI_FC1_JIT_MODULE)
            and _fc1_common(
                x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, out, eps, ARCHS
            )
            and _shape(fc1_weight, (FC1_ROWS, HIDDEN), torch.bfloat16)
            and fc1_weight.device == x.device
            and _opt(workspace, (x.shape[0], HIDDEN), torch.bfloat16, x.device)
        )
    except Exception:
        return False


def minimax_h3_fc1_swiglu(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [M, 14336]``."""
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import minimax_h3_fc1_swiglu

    return minimax_h3_fc1_swiglu(
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


def supports_minimax_h3_fc1_swiglu_mxfp8(
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
    eps: float = EPS,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        return (
            flashinfer_module_available(FI_FC1_MODULE, FI_FC1_JIT_MODULE)
            and _fc1_common(
                x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, out, eps, ARCHS
            )
            and _shape(fc1_weight_q, (FC1_ROWS, HIDDEN), torch.float8_e4m3fn)
            and fc1_scale_tiles.dtype == torch.uint8
            and fc1_scale_tiles.numel() == MXFP8_FC1_SCALE_TILE_BYTES
            and fc1_scale_tiles.is_contiguous()
            and fc1_weight_q.device == x.device
            and fc1_scale_tiles.device == x.device
            and _opt(workspace_q, (x.shape[0], HIDDEN), torch.float8_e4m3fn, x.device)
            and _opt_min_bytes(
                workspace_sf,
                mxfp8_fc1_activation_scale_workspace_bytes(x.shape[0]),
                x.device,
            )
        )
    except Exception:
        return False


def minimax_h3_fc1_swiglu_mxfp8(
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
    eps: float = EPS,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [M, 14336]``.

    ``workspace_sf`` is ``torch.zeros``-allocated per call when omitted:
    supply ``workspace_q``, ``workspace_sf`` and ``out`` for CUDA graphs.
    """
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
        minimax_h3_fc1_swiglu_mxfp8,
    )

    return minimax_h3_fc1_swiglu_mxfp8(
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


def supports_minimax_h3_fc1_swiglu_nvfp4(
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
    eps: float = EPS,
) -> bool:
    """Admission check mirroring the FlashInfer dispatcher; never raises.

    On cc 10.0/10.3 ``fc1_scale_tiles`` must be the SM100/103 combined tiles
    (11,010,048 bytes) and ``eps == 1e-5``; on cc 12.x the SM120 route accepts
    ``fc1_scale_tiles`` of ``28672 * 336`` bytes (``prepare_..._nvfp4_sm120``),
    any positive ``eps`` and a dense ``workspace_sf`` of ``M * 336`` bytes.
    """
    import torch

    from sglang.kernels.cake_kernels._support import device_capability

    try:
        if not flashinfer_module_available(FI_FC1_MODULE, FI_FC1_JIT_MODULE):
            return False
        if not cuda_tensor_on(x, NVFP4_FC1_DISPATCH_ARCHS):
            return False
        sm120 = device_capability(x.device.index)[0] == 12
        if sm120 and not flashinfer_module_available(FI_SM120_FC1_JIT_MODULE):
            return False
        if sm120:
            # The SM120 route accepts any positive finite eps.
            eps_ok = float(eps) > 0.0 and float(eps) == float(eps)
            base = _fc1_common(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                out,
                EPS,
                NVFP4_FC1_DISPATCH_ARCHS,
            )
            tiles = FC1_ROWS * (HIDDEN // 16)
            sf_bytes = x.shape[0] * (HIDDEN // 16)
            scale_ok = (
                isinstance(a_global_scale, torch.Tensor)
                and a_global_scale.numel() == 1
                and a_global_scale.dtype == torch.float32
                and a_global_scale.device == x.device
            )
        else:
            eps_ok = True
            base = _fc1_common(
                x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, out, eps, ARCHS
            )
            tiles = NVFP4_FC1_SCALE_TILE_BYTES
            sf_bytes = nvfp4_fc1_activation_scale_workspace_bytes(x.shape[0])
            scale_ok = _f32_scalar(a_global_scale, x.device)
        return (
            base
            and eps_ok
            and scale_ok
            and _f32_scalar(alpha, x.device)
            and _shape(fc1_weight_q, (FC1_ROWS, HIDDEN // 2), torch.uint8)
            and fc1_scale_tiles.dtype == torch.uint8
            and fc1_scale_tiles.numel() == tiles
            and fc1_scale_tiles.is_contiguous()
            and fc1_weight_q.device == x.device
            and fc1_scale_tiles.device == x.device
            and _opt(workspace_q, (x.shape[0], HIDDEN // 2), torch.uint8, x.device)
            and _opt_min_bytes(workspace_sf, sf_bytes, x.device)
        )
    except Exception:
        return False


def minimax_h3_fc1_swiglu_nvfp4(
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
    eps: float = EPS,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [M, 14336]``.

    Arch dispatcher: tcgen05 route on cc 10.0/10.3, ``mma.sync`` SM120 route on
    cc 12.x (which reads a tensor ``alpha`` with a host sync; pass a float
    there). ``workspace_sf`` is zero-allocated per call when omitted.
    """
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
        minimax_h3_fc1_swiglu_nvfp4,
    )

    return minimax_h3_fc1_swiglu_nvfp4(
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


def supports_prepare_minimax_h3_fc1_weight(fc1_weight: torch.Tensor) -> bool:
    """Admission for both FC1 weight preparations; never raises.

    Pure-torch / FlashInfer-quantizer work on any CUDA device; the produced
    layout is for the SM100/103 route (``prepare_minimax_h3_fc1_weight_nvfp4``
    returns the SM120 layout instead when ``fc1_weight`` lives on a cc 12.x
    device).
    """
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


def prepare_minimax_h3_fc1_weight_mxfp8(
    fc1_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(fc1_weight_q E4M3 [28672, 5376], fc1_scale_tiles uint8)``."""
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
        prepare_minimax_h3_fc1_weight_mxfp8,
    )

    return prepare_minimax_h3_fc1_weight_mxfp8(fc1_weight)


def prepare_minimax_h3_fc1_weight_nvfp4(
    fc1_weight: torch.Tensor, w_global_scale: Union[torch.Tensor, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(fc1_weight_q uint8 [28672, 2688], fc1_scale_tiles uint8)``.

    Arch dispatcher: SM120 layout when ``fc1_weight`` is on a cc 12.x device,
    SM100/103 combined tiles otherwise. Prepare on the arch that will run.
    """
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
        prepare_minimax_h3_fc1_weight_nvfp4,
    )

    return prepare_minimax_h3_fc1_weight_nvfp4(fc1_weight, w_global_scale)


def minimax_h3_nvfp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Host helper: f32 ``[1]`` NVFP4 global encode scale ``448 * 6 / absmax(tensor)``."""
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
        minimax_h3_nvfp4_global_scale,
    )

    return minimax_h3_nvfp4_global_scale(tensor)


def minimax_h3_nvfp4_alpha(
    a_global_scale: torch.Tensor, w_global_scale: torch.Tensor
) -> torch.Tensor:
    """Host helper: f32 ``[1]`` GEMM ``alpha = 1 / (a_gs * w_gs)``."""
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import minimax_h3_nvfp4_alpha

    return minimax_h3_nvfp4_alpha(a_global_scale, w_global_scale)


# ---------------------------------------------------------------------------
# Out-projection (D34 - D39)
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
        and attn_out.ndim == 4
        and attn_out.dtype == torch.bfloat16
        and attn_out.is_contiguous()
    ):
        return False
    p, m = int(attn_out.shape[0]), int(attn_out.shape[1])
    bf16 = torch.bfloat16
    return (
        p in SEQUENCE_PARALLEL_DEGREES
        and 1 <= m <= MAX_ROWS
        and tuple(attn_out.shape) == (p, m, NUM_HEADS // p, HEAD_DIM)
        and _table(gate, attn_out.device)
        # The epilogue forms the table offset as a 32x32 -> 64-bit multiply of
        # the row index and the row pitch: rows must not overlap and the pitch
        # must fit 32 bits.
        and HIDDEN <= gate.stride(0) < 2**32
        and _shape(gate_index, (m,), torch.int64)
        and _shape(residual, (m, HIDDEN), bf16)
        and gate_index.device == attn_out.device
        and residual.device == attn_out.device
        and _opt(out, (m, HIDDEN), bf16, attn_out.device)
    )


def supports_minimax_h3_out_proj(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        return (
            flashinfer_module_available(FI_OUT_PROJ_MODULE, FI_OUT_PROJ_JIT_MODULE)
            and _out_proj_common(attn_out, gate, gate_index, residual, out)
            and _shape(o_weight, (HIDDEN, ATTN_DIM), torch.bfloat16)
            and o_weight.device == attn_out.device
        )
    except Exception:
        return False


def minimax_h3_out_proj(
    attn_out: torch.Tensor,
    o_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [M, 5376]``."""
    from flashinfer.diffusion_ops.minimax_h3_out_proj import minimax_h3_out_proj

    return minimax_h3_out_proj(attn_out, o_weight, gate, gate_index, residual, out=out)


def supports_minimax_h3_out_proj_mxfp8(
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
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        m = int(attn_out.shape[1]) if attn_out.ndim == 4 else 0
        return (
            flashinfer_module_available(FI_OUT_PROJ_MODULE, FI_OUT_PROJ_JIT_MODULE)
            and _out_proj_common(attn_out, gate, gate_index, residual, out)
            and _shape(o_weight_q, (HIDDEN, ATTN_DIM), torch.float8_e4m3fn)
            and o_scale_tiles.dtype == torch.uint8
            and o_scale_tiles.numel() == MXFP8_O_SCALE_TILE_BYTES
            and o_scale_tiles.is_contiguous()
            and o_weight_q.device == attn_out.device
            and o_scale_tiles.device == attn_out.device
            and _opt(workspace_q, (m, ATTN_DIM), torch.float8_e4m3fn, attn_out.device)
            and _opt_min_bytes(
                workspace_sf,
                mxfp8_out_proj_activation_scale_workspace_bytes(m),
                attn_out.device,
            )
        )
    except Exception:
        return False


def minimax_h3_out_proj_mxfp8(
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
    """Forward to FlashInfer; returns BF16 ``out [M, 5376]``.

    ``workspace_sf`` is zero-allocated per call when omitted: supply the
    workspaces and ``out`` for CUDA graphs.
    """
    from flashinfer.diffusion_ops.minimax_h3_out_proj import minimax_h3_out_proj_mxfp8

    return minimax_h3_out_proj_mxfp8(
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


def supports_minimax_h3_out_proj_nvfp4(
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
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        m = int(attn_out.shape[1]) if attn_out.ndim == 4 else 0
        return (
            flashinfer_module_available(FI_OUT_PROJ_MODULE, FI_OUT_PROJ_JIT_MODULE)
            and _out_proj_common(attn_out, gate, gate_index, residual, out)
            and _f32_scalar(a_global_scale, attn_out.device)
            and _f32_scalar(alpha, attn_out.device)
            and _shape(o_weight_q, (HIDDEN, ATTN_DIM // 2), torch.uint8)
            and o_scale_tiles.dtype == torch.uint8
            and o_scale_tiles.numel() == NVFP4_O_SCALE_TILE_BYTES
            and o_scale_tiles.is_contiguous()
            and o_weight_q.device == attn_out.device
            and o_scale_tiles.device == attn_out.device
            and _opt(workspace_q, (m, ATTN_DIM // 2), torch.uint8, attn_out.device)
            and _opt_min_bytes(
                workspace_sf,
                nvfp4_out_proj_activation_scale_workspace_bytes(m),
                attn_out.device,
            )
        )
    except Exception:
        return False


def minimax_h3_out_proj_nvfp4(
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
    """Forward to FlashInfer; returns BF16 ``out [M, 5376]``.

    ``a_global_scale`` / ``alpha`` are moved to device f32 ``[1]`` by FlashInfer
    (a copy when given on CPU). ``workspace_sf`` is zero-allocated per call when
    omitted.
    """
    from flashinfer.diffusion_ops.minimax_h3_out_proj import minimax_h3_out_proj_nvfp4

    return minimax_h3_out_proj_nvfp4(
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


def supports_prepare_minimax_h3_o_weight(o_weight: torch.Tensor) -> bool:
    """Admission for both out-proj weight preparations; never raises."""
    import torch

    try:
        return (
            flashinfer_module_available(FI_OUT_PROJ_MODULE)
            and o_weight.is_cuda
            and torch.version.cuda is not None
            and _shape(o_weight, (HIDDEN, ATTN_DIM), torch.bfloat16)
        )
    except Exception:
        return False


def prepare_minimax_h3_o_weight_mxfp8(
    o_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(o_weight_q E4M3 [5376, 7168], o_scale_tiles uint8)``."""
    from flashinfer.diffusion_ops.minimax_h3_out_proj import (
        prepare_minimax_h3_o_weight_mxfp8,
    )

    return prepare_minimax_h3_o_weight_mxfp8(o_weight)


def prepare_minimax_h3_o_weight_nvfp4(
    o_weight: torch.Tensor, w_global_scale: Union[torch.Tensor, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(o_weight_q uint8 [5376, 3584], o_scale_tiles uint8)``."""
    from flashinfer.diffusion_ops.minimax_h3_out_proj import (
        prepare_minimax_h3_o_weight_nvfp4,
    )

    return prepare_minimax_h3_o_weight_nvfp4(o_weight, w_global_scale)
