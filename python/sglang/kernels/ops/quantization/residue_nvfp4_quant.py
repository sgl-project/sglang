"""Residue NVFP4 activation quantization (mext_r1) public wrappers.

mext_r1 IS residue ratio 1.0 -- the storage-efficient implementation of it,
not a mode competing with the ratio options. Every input channel is salient,
so instead of duplicating the weight:

    K-ext at ratio 1.0:  [x_q | r_q] @ [W | W]^T = x_q@W^T + r_q@W^T
    mext_r1:             rows (x_q, r_q) @ W^T, summed in the epilogue
                                                = (x_q + r_q)@W^T

the weight is stored once at K and the token rows are doubled. Consequently
this entrypoint takes no salient indices: there is no per-channel selection
to make on this path.

Kernel output:
  - data:  [2M, K/2] packed uint8 (two FP4 nibbles per byte), or [M, K] for
    the concat_k layout ([M, 2K] logical geometry)
  - scale: fp8-e4m3 bytes in the swizzled 128x4 tiled layout over the output
    geometry

``input_scale`` is a single global scale reciprocal shared by the base and
residue halves.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# elts mode: auto resolves per-arch in the C++ launcher -- datacenter
# Blackwell (cc 10.x) uses the pack16 path (per-thread SF vector + 256-bit
# loads), SM120 keeps pack8.
MEXT_R1_ELTS_AUTO = 0
MEXT_R1_ELTS_8 = 8
MEXT_R1_ELTS_16 = 16

# Activation layouts emitted by the quant kernel.
MEXT_R1_LAYOUT_CONCAT = 0  # [x rows... | r rows...]
MEXT_R1_LAYOUT_ROW_PAIR = 1  # [x0, r0, x1, r1, ...]
# K-concat: [M, 2K] geometry -- base in element cols [0, K), residue in
# [K, 2K), data AND SF in the canonical order of the (M, 2K) grid. A
# downstream GEMM cannot tell it from a stock quantization of a 2K-wide input.
MEXT_R1_LAYOUT_CONCAT_K = 3

_LAYOUT_NAMES = {
    "concat": MEXT_R1_LAYOUT_CONCAT,
    "row_pair": MEXT_R1_LAYOUT_ROW_PAIR,
    "concat_k": MEXT_R1_LAYOUT_CONCAT_K,
}


def _resolve_elts_mode(elts_mode: Union[int, str, None]) -> int:
    if elts_mode is None:
        # Perf knob only: forces a pack; numerics are identical either way.
        elts_mode = os.environ.get("SGLANG_RESIDUE_NVFP4_ELTS_PER_THREAD", "auto")
    if isinstance(elts_mode, str):
        normalized = elts_mode.strip().lower()
        if normalized in ("", "auto", "default"):
            return MEXT_R1_ELTS_AUTO
        if normalized == "8":
            return MEXT_R1_ELTS_8
        if normalized == "16":
            return MEXT_R1_ELTS_16
        raise ValueError(
            f"Unsupported mext_r1 elts_mode {elts_mode!r}. Expected: auto, 8, 16."
        )
    if elts_mode in (MEXT_R1_ELTS_AUTO, MEXT_R1_ELTS_8, MEXT_R1_ELTS_16):
        return int(elts_mode)
    raise ValueError(
        f"Unsupported mext_r1 elts_mode {elts_mode!r}. Expected: auto (0), 8, 16."
    )


def _resolve_layout_mode(layout_mode: Union[int, str]) -> int:
    if isinstance(layout_mode, str):
        normalized = layout_mode.strip().lower().replace("-", "_")
        if normalized in _LAYOUT_NAMES:
            return _LAYOUT_NAMES[normalized]
        raise ValueError(
            f"Unsupported mext_r1 layout_mode {layout_mode!r}. "
            f"Expected one of: {', '.join(_LAYOUT_NAMES)}."
        )
    if layout_mode in _LAYOUT_NAMES.values():
        return int(layout_mode)
    raise ValueError(
        f"Unsupported mext_r1 layout_mode {layout_mode!r}. "
        f"Expected one of: {', '.join(_LAYOUT_NAMES)} (0-3)."
    )


@cache_once
def _jit_scaled_fp4_quant_mext_r1_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "residue_nvfp4_scaled_fp4_quant_mext_r1",
        *args,
        cuda_files=["residue_nvfp4/residue_nvfp4_quant.cuh"],
        cuda_wrappers=[
            ("scaled_fp4_quant_mext_r1", f"scaled_fp4_quant_mext_r1<{args}>")
        ],
    )


def _mext_r1_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    elts_mode: int,
    layout_mode: int,
    output_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = input.shape
    del m
    if layout_mode == MEXT_R1_LAYOUT_CONCAT_K:
        data = input.new_empty((output_m, k), dtype=torch.uint8)
        num_m_tiles = (output_m + 127) // 128
        num_k_tiles = (2 * k + 63) // 64
    else:
        data = input.new_empty((2 * output_m, k // 2), dtype=torch.uint8)
        num_m_tiles = (2 * output_m + 127) // 128
        num_k_tiles = (k + 63) // 64
    scale = input.new_empty(num_m_tiles * num_k_tiles * 512, dtype=torch.uint8)
    return data, scale


@register_custom_op(
    op_name="residue_nvfp4_scaled_fp4_quant_mext_r1",
    mutates_args=[],
    fake_impl=_mext_r1_fake,
)
def _scaled_fp4_quant_mext_r1_op(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    elts_mode: int,
    layout_mode: int,
    output_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = input.shape

    output = torch.empty(
        (2 * output_m) * k // 2, dtype=torch.uint8, device=input.device
    )
    if layout_mode == MEXT_R1_LAYOUT_CONCAT_K:
        # (M, 2K) grid: same data bytes, but the SF atom tiling follows the
        # output geometry, not the row-doubled one.
        num_m_tiles = (output_m + 127) // 128
        num_k_tiles = (2 * k + 63) // 64
    else:
        num_m_tiles = (2 * output_m + 127) // 128
        num_k_tiles = (k + 63) // 64
    output_scale = torch.empty(
        num_m_tiles * num_k_tiles * 512, dtype=torch.uint8, device=input.device
    )

    module = _jit_scaled_fp4_quant_mext_r1_module(input.dtype)
    # output_m is intentionally NOT passed to the kernel: the C++ op derives
    # it from input.shape so the launch can never desync from the buffers
    # under torch.compile shape specialization.
    module.scaled_fp4_quant_mext_r1(
        input.contiguous(),
        output,
        output_scale,
        input_scale,
        elts_mode,
        layout_mode,
    )

    if layout_mode == MEXT_R1_LAYOUT_CONCAT_K:
        # [M, 2K] packed: K bytes per row (2 FP4 elements per byte).
        return output.view(output_m, k), output_scale
    return output.view(2 * output_m, k // 2), output_scale


def scaled_fp4_quant_mext_r1(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    elts_mode: Union[int, str, None] = None,
    layout_mode: Union[int, str] = "concat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ratio-1.0 M-extension FP4 activation quantization.

    Args:
        input: [M, K] fp16/bf16 CUDA tensor, K % 16 == 0.
        input_scale: scalar or 1-D float32 CUDA tensor with 1 element (global
            scale reciprocal).
        elts_mode: "auto" (default), 8, or 16 elements per thread.
        layout_mode: "concat", "row_pair", or "concat_k".

    Returns:
        (data, scale): packed FP4 data ([2*output_M, K/2] uint8, or
        [output_M, K] for concat_k) and the swizzled fp8-e4m3 scale bytes.
    """
    resolved_elts = _resolve_elts_mode(elts_mode)
    resolved_layout = _resolve_layout_mode(layout_mode)

    m, k = input.shape
    if k % 16 != 0:
        raise ValueError(f"scaled_fp4_quant_mext_r1 requires K % 16 == 0, got {k}")

    if input_scale.ndim == 0:
        # ModelOpt stores per-tensor activation scales as scalar parameters,
        # while the JIT FFI schema deliberately uses a 1-D tensor.
        input_scale = input_scale.reshape(1)
    elif input_scale.ndim != 1:
        raise ValueError(
            "scaled_fp4_quant_mext_r1 requires input_scale to be scalar or "
            f"1-D, got shape {tuple(input_scale.shape)}"
        )
    if input_scale.numel() != 1:
        raise ValueError(
            "scaled_fp4_quant_mext_r1 requires input_scale with exactly 1 "
            f"element, got {input_scale.numel()}"
        )

    return _scaled_fp4_quant_mext_r1_op(
        input, input_scale, resolved_elts, resolved_layout, m
    )


# ── k_ext (selective residue, ratios 1/8, 2/8, 4/8) ────────────────────────

# ratio -> salient channels per 8-channel block ("residue_per_8").
# Ratio 1.0 is deliberately absent: K-extension at ratio 1.0 would store the
# weight twice; full-rank residue is served by mext_r1 instead.
SUPPORTED_K_EXT_RATIOS = {
    0.125: 1,
    0.25: 2,
    0.5: 4,
}


def _resolve_residue_per_8(num_salient: int, k: int) -> int:
    ratio = num_salient / k
    for supported_ratio, residue_per_8 in SUPPORTED_K_EXT_RATIOS.items():
        if abs(ratio - supported_ratio) < 1e-3:
            return residue_per_8
    supported = ", ".join(f"{r:g}" for r in sorted(SUPPORTED_K_EXT_RATIOS))
    raise ValueError(
        f"Unsupported salient ratio {ratio:.4f} ({num_salient}/{k}). "
        f"Supported k_ext ratios: {supported}; ratio 1.0 is mext_r1."
    )


def indices_to_channel_masks(
    salient_indices: torch.Tensor, num_channels: int
) -> torch.Tensor:
    """Per-8-channel salient bitmask consumed by the k_ext quant kernels.

    Byte i's bit b marks channel 8*i+b salient. The exporter selects the same
    count in every 8-channel block, which the kernels assume.
    """
    assert num_channels % 8 == 0, "num_channels must be a multiple of 8"
    num_masks = num_channels // 8
    channel_masks = torch.zeros(
        num_masks, dtype=torch.uint8, device=salient_indices.device
    )
    mask_idx = salient_indices // 8
    bit_pos = salient_indices % 8
    bit_masks = (1 << bit_pos).to(torch.uint8)
    channel_masks.index_add_(0, mask_idx, bit_masks)
    return channel_masks


@cache_once
def _jit_scaled_fp4_quant_with_mask_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "residue_nvfp4_scaled_fp4_quant_with_mask",
        *args,
        cuda_files=["residue_nvfp4/residue_nvfp4_kext_quant.cuh"],
        cuda_wrappers=[
            ("scaled_fp4_quant_with_mask", f"scaled_fp4_quant_with_mask<{args}>")
        ],
    )


def _kext_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    n_ext: int,
    residue_per_8: int,
    elts_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = input.shape[0]
    data = input.new_empty((m, n_ext // 2), dtype=torch.uint8)
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (n_ext + 63) // 64
    scale = input.new_empty(num_m_tiles * num_k_tiles * 512, dtype=torch.uint8)
    return data, scale


@register_custom_op(
    op_name="residue_nvfp4_scaled_fp4_quant_with_mask",
    mutates_args=[],
    fake_impl=_kext_fake,
)
def _scaled_fp4_quant_with_mask_op(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    n_ext: int,
    residue_per_8: int,
    elts_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = input.shape
    output = torch.empty(m * n_ext // 2, dtype=torch.uint8, device=input.device)
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (n_ext + 63) // 64
    output_scale = torch.empty(
        num_m_tiles * num_k_tiles * 512, dtype=torch.uint8, device=input.device
    )

    module = _jit_scaled_fp4_quant_with_mask_module(input.dtype)
    module.scaled_fp4_quant_with_mask(
        input.contiguous(),
        output,
        output_scale,
        input_scale,
        channel_masks,
        n_ext,
        residue_per_8,
        elts_mode,
    )
    return output.view(m, n_ext // 2), output_scale


def scaled_fp4_quant_with_mask(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    num_salient: int,
    *,
    elts_mode: Union[int, str, None] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k_ext FP4 activation quantization with precomputed channel masks.

    Quantizes the full row to NVFP4 and appends the quantized dequantization
    residues of the salient channels, producing the extended activation the
    K-extended weight consumes.

    Args:
        input: [M, K] fp16/bf16 CUDA tensor, K % 16 == 0.
        input_scale: scalar or 1-D float32 CUDA tensor with 1 element (global
            scale reciprocal).
        channel_masks: uint8 [K/8] bitmask (see indices_to_channel_masks).
        num_salient: salient channel count; num_salient/K must be a supported
            ratio (1/8, 2/8, 4/8).
        elts_mode: "auto" (default; B200-measured pack16 policy on cc 10.x),
            8, or 16.

    Returns:
        (data, scale): [M, (K+num_salient)/2] packed uint8 rows laid out as
        [base | residue], and the swizzled fp8-e4m3 scale bytes over
        (M, K+num_salient).
    """
    m, k = input.shape
    del m
    if input_scale.ndim == 0:
        # ModelOpt stores per-tensor activation scales as scalar parameters,
        # while the JIT FFI schema uses a 1-D tensor. Keep this normalization
        # aligned with scaled_fp4_quant_mext_r1 above.
        input_scale = input_scale.reshape(1)
    elif input_scale.ndim != 1:
        raise ValueError(
            "scaled_fp4_quant_with_mask requires input_scale to be scalar or "
            f"1-D, got shape {tuple(input_scale.shape)}"
        )
    if input_scale.numel() != 1:
        raise ValueError(
            "scaled_fp4_quant_with_mask requires input_scale with 1 element, "
            f"got {input_scale.numel()}"
        )

    n_ext = k + int(num_salient)
    residue_per_8 = _resolve_residue_per_8(int(num_salient), k)
    resolved_elts = _resolve_elts_mode(elts_mode)
    return _scaled_fp4_quant_with_mask_op(
        input, input_scale, channel_masks, n_ext, residue_per_8, resolved_elts
    )
