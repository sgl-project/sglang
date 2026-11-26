"""MXFP8 weight-scale layout for inkling2's SM120 CuTe grouped GEMM
(sglang.kernels.ops.gemm.inkling2_grouped_sm120.mxfp8_expert_matmul)."""

import torch
import triton


def mxfp8_weight_scales(scales, out=None):
    """(experts, n, k // 32) UE8M0 weight scales as the CuTe GEMM's (experts, n,
    k // 128) MN-major int32 words of four K blocks, written into out when given."""
    experts, rows, blocks = scales.shape
    words = triton.cdiv(blocks, 4)
    padded = scales.new_full((experts, rows, words * 4), 127)
    padded[..., :blocks] = scales
    if out is None:
        out = torch.empty(
            (experts, words, rows), dtype=torch.int32, device=scales.device
        ).transpose(1, 2)
    return out.copy_(padded.view(torch.int32))
