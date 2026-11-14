# SPDX-License-Identifier: Apache-2.0

"""HALOps w8a8 MXFP8 dense linear runner (Blackwell SM100)."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.quantization.halops_mxfp8 import (
    HALOPS_BIAS_M_ALIGN,
    HALOPS_K_ALIGN,
    HALOPS_MODE,
    HALOPS_SFATOM_ROW_ALIGN,
    load_halops_gemm,
)
from sglang.srt.runtime_context import get_platform


def halops_tile_available(*, m: int, n: int, has_bias: bool) -> bool:
    """Whether a HALOps w8a8-mxfp8 tile accepts this shape.

    N must be 128-aligned for the SfAtom weight-scale atoms. Any M works without
    a bias; the bias epilogue has no per-row store predicate, so a bias layer
    needs M to be a whole multiple of 256, and anything else goes to the fallback
    rather than raising inside HALOps.
    """
    if n % HALOPS_SFATOM_ROW_ALIGN != 0:
        return False
    return not has_bias or m % HALOPS_BIAS_M_ALIGN == 0


def _flashinfer_fallback(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_fallback: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Serve an MXFP8 linear no HALOps tile can take, via FlashInfer CUTLASS.

    FlashInfer consumes the interleaved scale layout, not the canonical row-major
    one the HALOps load path keeps in ``weight_scale_inv``, so swizzle here: this
    is the cold path by construction and the scale is ``[N, K // 32]`` bytes.
    """
    if weight_scale_fallback is None:
        raise RuntimeError(
            "HALOps MXFP8 has no tile for this GEMM and no canonical UE8M0 weight "
            "scale was kept for the FlashInfer fallback. Re-run with "
            "--fp8-gemm-backend=flashinfer_cutlass."
        )

    from flashinfer import block_scale_interleave

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    # FlashInfer's activation quantizer only takes bf16/fp16; fp32 is one of the
    # reasons we are on this path at all, so downcast and restore the caller's dtype.
    orig_dtype = input.dtype
    if orig_dtype == torch.float32:
        input = input.to(torch.bfloat16)

    # 1D on purpose: flashinfer_mxfp8_blockscaled_linear transposes a 2D scale, and
    # block_scale_interleave already returns the flat interleaved form.
    weight_scale = block_scale_interleave(
        weight_scale_fallback.contiguous()
    ).contiguous()
    output = flashinfer_mxfp8_blockscaled_linear(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        input_scale=None,
        bias=bias,
    )
    return output if output.dtype == orig_dtype else output.to(orig_dtype)


def halops_w8a8_mxfp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    weight_scale_fallback: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MXFP8 dense linear via HALOps ``dense_gemm`` (Blackwell SM100).

    ``weight`` is fp8_e4m3 ``[N, K]`` and ``weight_scale`` is in HALOps' SfAtom
    layout (prepared at load time). Signature mirrors
    ``_deepgemm_w8a8_mxfp8_linear_with_fallback``.

    ``input_scale`` has no supported producer for this backend, so a non-None
    value is rejected rather than interpreted.
    """
    n, k = weight.shape
    m = input.numel() // input.shape[-1]

    if input_scale is not None:
        # No producer reaches here with one: Mxfp8SwizzledInput is rejected for a
        # non-FlashInfer backend upstream, and the (q, scale) producers are
        # DeepGEMM-only and carry a packed int32 scale. Guessing the layout would
        # silently mis-swizzle the scale.
        raise NotImplementedError(
            "HALOps MXFP8 takes bf16/fp16 activations only, not an already "
            "quantized (fp8 input, UE8M0 scale) pair. Re-run with "
            "--fp8-gemm-backend=flashinfer_cutlass or deep_gemm."
        )

    if not (
        halops_tile_available(m=m, n=n, has_bias=bias is not None)
        and k % HALOPS_K_ALIGN == 0
        and input.dtype in (torch.bfloat16, torch.float16)
        and get_platform().is_sm100
    ):
        return _flashinfer_fallback(
            input=input,
            weight=weight,
            weight_scale_fallback=weight_scale_fallback,
            bias=bias,
        )

    dense_gemm, group_wise_quant = load_halops_gemm()

    # .contiguous() is required, not defensive: dense_gemm and group_wise_quant read
    # raw row-major pointers, so a strided view silently produces wrong numerics.
    input_2d = input.reshape(-1, k).contiguous()

    q_input, sfa = group_wise_quant(input_2d, mode=HALOPS_MODE, use_swizzle=True)

    if bias is not None:
        # dense_gemm types the bias as the (always bf16) output element type.
        bias = bias.to(torch.bfloat16)

    # dense_gemm only writes bf16; cast back below for an fp16 layer.
    output = torch.empty((m, n), dtype=torch.bfloat16, device=input_2d.device)

    dense_gemm(
        input=q_input,
        weight=weight,
        sfa=sfa,
        sfb=weight_scale,
        output=output,
        bias=bias,
        alpha=None,
        sm_count=-1,
        mode=HALOPS_MODE,
    )

    return output.to(input.dtype).reshape(*input.shape[:-1], n)
