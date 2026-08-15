"""Fused pack for the trtllm routed-MoE topk format.

The trtllm routed MoE consumes top-k routing as a single int32 per (token, slot):
``PackedScoreIdx`` = ``(expert_id << 16) | bf16_weight_bits`` (little-endian: low 16
bits = bf16 weight, high 16 bits = int16 expert id).

The torch reference builds this with a cluster of ~4 elementwise ops (cast, cast, view,
bitshift, or). On the decode LoRA path these run on the main CUDA stream between
``per_token_group_quant_fp8`` and the trtllm MoE op. This kernel collapses them into a
single Triton launch. Bit-identical to the torch reference: weights are softmax/renormalized
(>= 0 -> bf16 sign bit is 0, so the low 16 bits never collide with the id field and masking
== torch's sign-extend-then-or); expert ids are small (< num_experts).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_topk_kernel(ids_ptr, w_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ids = tl.load(ids_ptr + offs, mask=mask)  # int32
    w = tl.load(w_ptr + offs, mask=mask)  # float32
    wb = w.to(tl.bfloat16)
    wbits = wb.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    packed = (ids << 16) | wbits
    tl.store(out_ptr + offs, packed, mask=mask)


def fused_pack_topk(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Single-launch replacement for the elementwise routed-MoE topk pack.

    Returns int32 ``[*, top_k]`` packed tensor, bit-identical to the torch reference.
    """
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    topk_weights = topk_weights.contiguous()
    out = torch.empty_like(topk_ids, dtype=torch.int32)
    numel = out.numel()
    if numel == 0:
        return out
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _pack_topk_kernel[grid](topk_ids, topk_weights, out, numel, BLOCK=BLOCK)
    return out
