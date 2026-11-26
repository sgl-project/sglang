"""SM120 routed quantization of expert-sorted rows (offsets[0] == 0): the kernels of
routed_quant_sm120.cuh quantize the rows for FlashInfer's SM120 grouped GEMMs
(sglang.kernels.ops.gemm.inkling2_grouped_sm120), CUTLASS for NVFP4 and CuTe for
MXFP8. The swiglu kernels read those GEMMs' BF16-rounded raw accumulators and apply
the NVFP4 per-row outer scales."""

import torch
import triton

from sglang.kernels.jit.utils import cache_once, load_jit

_KERNELS = (
    "nvfp4_quantize",
    "nvfp4_swiglu_quantize",
    "mxfp8_quantize",
    "mxfp8_swiglu_quantize",
)
# Every width compiles all four kernels, and the NVFP4 ones need split(width) > 1,
# so width <= 8192; offsets cover at most kMaxExperts experts (routed_quant.cuh).
_MAX_WIDTH = 8192
_MAX_EXPERTS = 1024


def can_quantize(widths, experts: int) -> bool:
    return experts <= _MAX_EXPERTS and all(width <= _MAX_WIDTH for width in widths)


@cache_once
def _module(width: int):
    return load_jit(
        "inkling2_routed_quant_sm120",
        str(width),
        cuda_files=["inkling2/routed_quant_sm120.cuh"],
        cuda_wrappers=[
            (name, f"inkling2_routed_quant_sm120::{name}<{width}>") for name in _KERNELS
        ],
    )


def _sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _nvfp4_outputs(rows, width, experts, device):
    """Packed values, scales in 128-row blocks per expert, and per-row amax."""
    scale_rows = triton.cdiv(rows + 127 * experts, 128) * 128
    return (
        torch.empty((rows, width // 2), dtype=torch.uint8, device=device),
        torch.empty((scale_rows, width // 16), dtype=torch.uint8, device=device),
        torch.empty(rows, dtype=torch.float32, device=device),
    )


def nvfp4_routed_quantize(x, dst2src, top_k, offsets):
    """NVFP4 values, scales and amax of the routed rows (row r reads token
    dst2src[r] // top_k) in [offsets[0], offsets[-1])."""
    width = x.shape[1]
    values, scales, amax = _nvfp4_outputs(
        rows=dst2src.numel(), width=width, experts=offsets.numel() - 1, device=x.device
    )
    if dst2src.numel():
        _module(width).nvfp4_quantize(
            x, dst2src, values, scales, amax, offsets, top_k, _sm_count(x.device)
        )
    return values, scales, amax


def nvfp4_routed_swiglu_quantize(
    acc, acc_amax, w_amax, w_denominator, gammas, offsets, gate_cap, up_cap
):
    """Scale the BF16 gate/up accumulator rows, apply the capped swiglu and
    quantize to NVFP4 for the next grouped GEMM."""
    rows, width = acc.shape[0], acc.shape[1] // 2
    values, scales, amax = _nvfp4_outputs(
        rows=rows, width=width, experts=offsets.numel() - 1, device=acc.device
    )
    if rows:
        _module(width).nvfp4_swiglu_quantize(
            acc,
            acc_amax,
            w_amax,
            w_denominator,
            gammas,
            values,
            scales,
            amax,
            offsets,
            gate_cap,
            up_cap,
            _sm_count(acc.device),
        )
    return values, scales, amax


def _mxfp8_outputs(rows, width, experts, device):
    """Values, zeroed CuTe scale storage (width // 128 words, m_padded rows), and
    m_padded."""
    m_padded = (rows + 3 * experts) // 4 * 4
    return (
        torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=device),
        torch.zeros((width // 128, m_padded), dtype=torch.int32, device=device),
        m_padded,
    )


def mxfp8_routed_quantize(x, dst2src, top_k, offsets):
    """MXFP8 values and CuTe scales of the routed rows (row r reads token
    dst2src[r] // top_k) in [offsets[0], offsets[-1])."""
    rows, width = dst2src.numel(), x.shape[1]
    values, scales, m_padded = _mxfp8_outputs(
        rows=rows, width=width, experts=offsets.numel() - 1, device=x.device
    )
    if rows:
        _module(width).mxfp8_quantize(
            x,
            dst2src,
            values.view(torch.uint8),
            scales,
            offsets,
            top_k,
            m_padded,
            _sm_count(x.device),
        )
    return values, scales.t()


def mxfp8_routed_swiglu_quantize(acc, gammas, offsets, gate_cap, up_cap):
    """Capped swiglu of the BF16 gate/up rows, rounded to BF16 and quantized to
    MXFP8 (values and CuTe scales) for the next grouped GEMM."""
    rows, width = acc.shape[0], acc.shape[1] // 2
    values, scales, m_padded = _mxfp8_outputs(
        rows=rows, width=width, experts=offsets.numel() - 1, device=acc.device
    )
    if rows:
        _module(width).mxfp8_swiglu_quantize(
            acc,
            gammas,
            values.view(torch.uint8),
            scales,
            offsets,
            m_padded,
            gate_cap,
            up_cap,
            _sm_count(acc.device),
        )
    return values, scales.t()
