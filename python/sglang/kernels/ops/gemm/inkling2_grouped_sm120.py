"""FlashInfer's SM120 block-scaled grouped GEMMs on inkling2's expert-sorted rows
(offsets[0] == 0), CUTLASS for NVFP4 and CuTe for MXFP8, with the activations
quantized by sglang.kernels.ops.moe.inkling2_experts_sm120. Outputs are BF16-rounded
raw accumulators; NVFP4 per-row outer scales are applied by the consumer."""

import torch


def _tiles(tokens, width):
    """(swap_ab, tile_n, tile_k), measured best on RTX 5090 for 128 experts, top-8,
    N 2304 and K (width) 1152 or 2304, with cold weights."""
    if tokens <= 128:
        return True, 32, 128
    if tokens <= 1536:
        return True, 32, 256
    if tokens <= 3072:
        return True, 64, 256
    return (False, 128, 256) if width >= 2048 else (True, 128, 128)


def nvfp4_expert_matmul(values, scales, weight, weight_scales, offsets, tokens):
    """BF16-rounded raw accumulators of the grouped NVFP4 GEMM of expert-sorted
    rows; weight_scales are in the swizzle_blockscale layout."""
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

    experts, n, packed_width = weight.shape
    out = torch.empty((values.shape[0], n), dtype=torch.bfloat16, device=values.device)
    if values.shape[0]:
        swap_ab, tile_n, tile_k = _tiles(tokens=tokens, width=packed_width * 2)
        group_gemm_nvfp4_nt_groupwise(
            values,
            weight,
            scales,
            weight_scales.view(torch.uint8).view(experts, n, packed_width // 8),
            offsets,
            tile_n=tile_n,
            tile_k=tile_k,
            swap_ab=swap_ab,
            out=out,
        )
    return out


def mxfp8_expert_matmul(values, scales, weight, weight_scales, offsets):
    """BF16 grouped MXFP8 GEMM of expert-sorted rows; weight_scales are
    mxfp8_weight_scales output."""
    from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise

    out = torch.empty(
        (values.shape[0], weight.shape[1]), dtype=torch.bfloat16, device=values.device
    )
    if values.shape[0]:
        moe_gemm_mxfp8_nt_groupwise(
            values,
            weight,
            scales,
            weight_scales,
            offsets,
            scale_granularity_mnk=(1, 1, 32),
            out=out,
        )
    return out
