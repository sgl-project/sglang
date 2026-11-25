"""Forward of the MXFP4 (W4A4) Quark experts, in Quark's order: both GEMM inputs are
fake-quantized to MXFP4, SwiGLU runs without the route weight, and the route weights are
applied in the FP32 top-k sum after w2 (the native path folds them in before w2)."""

import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation import situ_and_mul
from sglang.kernels.ops.moe import inkling2_experts_hip as experts_hip
from sglang.kernels.ops.moe import inkling2_quark_gemm_hip as gemm
from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.kernels.ops.quantization.mxfp4_quant_rows import mxfp4_quant_rows
from sglang.srt.layers.quantization.inkling2_quark import MXFP4_BLOCK_SIZE


def fake_quantize_mxfp4(x, pair_table):
    """BF16 [rows, K] -> the (exact) BF16 values of its per-row MXFP4 quantization.
    Equal to Quark's quantizer for finite blocks with amax < 1.75 * 2**127; larger
    blocks become zero (mxfp4_quant_rows flushes the denormal inverse scale 2**-127)."""
    packed, scale = mxfp4_quant_rows(x)
    values = F.embedding(packed.view(torch.uint8).int(), pair_table).view(
        x.shape[0], -1, MXFP4_BLOCK_SIZE
    )
    return (values * scale.to(torch.bfloat16).unsqueeze(-1)).view_as(x)


def _bf16_rows(partials):
    """One slab is already BF16; split-K FP32 slabs are summed and rounded once."""
    return partials[0] if partials.shape[0] == 1 else partials.sum(0).to(torch.bfloat16)


def forward_experts(experts, x, weights, ids, combine):
    """``ids`` and FP32 ``weights`` [tokens, top_k] come from the router."""
    tokens, top_k = ids.shape
    rows = ids.numel()
    num_local = experts.w13_weight.shape[0]
    start = experts.ep_rank * num_local
    num_experts = experts.num_global_experts
    # both GEMMs run with the same block height and share the routing schedule
    block_m = gemm.default_config_quark(gemm.MXFP4, rows, num_experts).bm
    route = experts_hip.route(
        ids, start, num_local, num_experts, weights, block_m=block_m
    )
    gate_up = _bf16_rows(
        gemm.expert_gemm_quark(
            fake_quantize_mxfp4(x.contiguous(), experts.pair_table),
            experts.w13_weight,
            experts.w13_scale,
            None,
            route,
            experts.output_sizes["w13"],
            top_k,
            torch.bfloat16,
            rows,
            fmt=gemm.MXFP4,
            num_experts=num_experts,
        )
    )
    # the export interleaves the columns (even = gate, odd = up); SiTU reads [gate | up]
    hidden = situ_and_mul(
        gate_up.view(rows, -1, 2).transpose(1, 2).reshape(rows, -1),
        None,
        experts.gate_cap,
        experts.up_cap,
    )
    down = _bf16_rows(
        gemm.expert_gemm_quark(
            fake_quantize_mxfp4(hidden, experts.pair_table),
            experts.w2_weight,
            experts.w2_scale,
            None,
            route,
            experts.output_sizes["w2"],
            1,
            torch.bfloat16,
            rows,
            fmt=gemm.MXFP4,
            num_experts=num_experts,
        )
    )
    local = None
    # skip with a negative id, not a zero weight: other ranks' rows are unset memory
    if experts.ep_size > 1:
        local = torch.where((ids >= start) & (ids < start + num_local), ids, -1)
    sums = moe_fused_mul_sum(
        down.view(tokens, top_k, -1),
        weights,
        torch.empty((tokens, down.shape[1]), dtype=torch.float32, device=x.device),
        topk_ids=local,
    )
    return experts.finish_forward(sums, combine, x.dtype)
