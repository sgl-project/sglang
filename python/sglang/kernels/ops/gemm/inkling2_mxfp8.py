import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.gemm.inkling2_nvfp4 import swizzle_block_scales
from sglang.kernels.ops.moe.inkling2_moe import (
    _sm_count,
    mxfp8_quantize,
    routed_quant_module,
)

PDL = is_arch_support_pdl()


@triton.jit
def _routed_swap_ab_matmul(
    X,
    XS,
    W,
    WS,
    Y,
    Counts,
    Offsets,
    Schedule,
    Dst2Src,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TOP_K: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    entry = tl.load(Schedule + tl.program_id(0))
    if entry == -1:
        return
    expert = (entry & 0xFFFF).to(tl.int64)
    block = (entry >> 16) * 128
    count = tl.minimum(tl.load(Counts + expert) - block, 128)
    col0 = tl.program_id(1) * BN
    cols = col0 + tl.arange(0, BN)
    for m0 in range(0, count, BM):
        local = block + m0 + tl.arange(0, BM)
        row_mask = m0 + tl.arange(0, BM) < count
        row_ids = tl.load(Offsets + expert).to(tl.int64) + local
        src_ids = row_ids
        if Dst2Src is not None:
            src_ids = (
                tl.load(Dst2Src + row_ids, row_mask, other=0).to(tl.int64) // TOP_K
            )
        acc = tl.zeros((BN, BM), tl.float32)
        for k0 in range(0, K, BK):
            offsets = k0 + tl.arange(0, BK)
            scales = k0 // 32 + tl.arange(0, BK // 32)
            w = W.load([(expert * N + col0).to(tl.int32), k0])
            ws = tl.load(
                WS + (expert * N + cols)[:, None] * (K // 32) + scales[None, :]
            )
            x = tl.load(
                X + src_ids[:, None] * K + offsets[None, :],
                row_mask[:, None],
                other=0.0,
            )
            xs = tl.load(
                XS + src_ids[:, None] * (K // 32) + scales[None, :],
                row_mask[:, None],
                other=0,
            )
            acc = tl.dot_scaled(
                w, ws, "e4m3", tl.trans(x), xs, "e4m3", acc, fast_math=True
            )
        tl.store(Y + row_ids[None, :] * N + cols[:, None], acc, row_mask[None, :])


def _routed_matmul(
    values, x_scales, rows, tokens, weight, scales, metadata, dst2src, top_k
):
    """Routed rows of the grouped MXFP8 GEMM as weight-rows x tokens tiles."""
    experts, output_size, width = weight.shape
    counts, offsets, schedule = metadata
    out = torch.empty((rows, output_size), dtype=torch.bfloat16, device=values.device)
    block_k = next(b for b in (512, 256, 128, 64, 32) if width % b == 0 and width > b)
    if rows:
        _routed_swap_ab_matmul[(min(schedule.numel(), rows), output_size // 128)](
            values,
            x_scales,
            TensorDescriptor.from_tensor(
                weight.view(experts * output_size, width), [128, block_k]
            ),
            scales,
            out,
            counts,
            offsets,
            schedule,
            dst2src,
            output_size,
            width,
            16 if tokens <= 16 else 32,
            128,
            block_k,
            top_k,
            num_warps=4,
            num_stages=2 if block_k == 512 else 3,
            launch_pdl=PDL,
        )
    return out


def routed_expert_matmul(x, dst2src, top_k, weight, scales, metadata):
    """Grouped MXFP8 GEMM over the routed token rows of x (row r reads token
    dst2src[r] // top_k), quantizing the token rows once."""
    values, x_scales = mxfp8_quantize(x.contiguous())
    return _routed_matmul(
        values,
        x_scales,
        dst2src.numel(),
        x.shape[0],
        weight,
        scales,
        metadata,
        dst2src,
        top_k,
    )


def _swiglu_quantize(gate_up, gammas, gate_cap, up_cap, offsets):
    """MXFP8 values and row-major scales of the capped swiglu of the gate_up rows."""
    rows, width = gate_up.shape[0], gate_up.shape[1] // 2
    values = torch.empty(
        (rows, width), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    x_scales = torch.empty(
        (rows, width // 32), dtype=torch.uint8, device=gate_up.device
    )
    if rows:
        routed_quant_module(width).mxfp8_swiglu_quantize(
            gate_up,
            gammas,
            values.view(torch.uint8),
            x_scales,
            offsets,
            gate_cap,
            up_cap,
            _sm_count(gate_up.device),
        )
    return values, x_scales


def routed_swiglu_expert_matmul(
    gate_up, gammas, gate_cap, up_cap, weight, scales, metadata, tokens
):
    """Grouped MXFP8 GEMM over the capped swiglu of the gate_up rows, with the
    swiglu fused into the activation quantization."""
    values, x_scales = _swiglu_quantize(gate_up, gammas, gate_cap, up_cap, metadata[1])
    return _routed_matmul(
        values, x_scales, gate_up.shape[0], tokens, weight, scales, metadata, None, 1
    )


def cutlass_weight_scales(scales: torch.Tensor, out: torch.Tensor | None = None):
    """(experts, n, k // 32) scales in the Blackwell block-scaled layout, written
    into out when given (in place, for CUDA-graph-held buffers)."""
    experts, n, groups = scales.shape
    size = triton.cdiv(n, 128) * 128 * triton.cdiv(groups * 32, 128) * 4
    if out is None:
        out = torch.zeros((experts, size), dtype=torch.uint8, device=scales.device)
    return swizzle_block_scales(scales, out)
