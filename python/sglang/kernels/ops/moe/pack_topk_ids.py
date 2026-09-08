"""Pack ``(topk_id, topk_weight)`` pairs into one int32 per entry.

Migrated from ``sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe``
(RFC #29630, Phase 2.5). Used by the FlashInfer TRT-LLM routed-MoE path, which
consumes routing ids and bf16 weights packed as ``(id << 16) | weight_bits``.
Routing ids may be int32 or the int64 dtype produced by ``torch.topk``; they are
converted to int32 by the Triton kernel before packing.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


class PackTopkIds:
    @classmethod
    def execute(
        cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        return cls.triton(topk_ids, topk_weights)

    @classmethod
    def vanilla(
        cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        weight_bits = (
            topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        )
        return (topk_ids.to(torch.int32) << 16) | weight_bits

    @classmethod
    def triton(cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        assert topk_ids.shape == topk_weights.shape, (
            f"shape mismatch: {topk_ids.shape=} vs {topk_weights.shape=}"
        )
        assert topk_ids.ndim >= 1, f"expected >=1D, got {topk_ids.shape=}"

        assert topk_ids.dtype in (
            torch.int32,
            torch.int64,
        ), f"topk_ids must be int32 or int64, got {topk_ids.dtype}"
        assert topk_weights.dtype == torch.float32, (
            f"topk_weights must be float32, got {topk_weights.dtype}"
        )

        assert topk_ids.is_contiguous(), "topk_ids must be contiguous"
        assert topk_weights.is_contiguous(), "topk_weights must be contiguous"

        out = torch.empty_like(topk_ids, dtype=torch.int32)
        numel = out.numel()
        if numel == 0:
            return out

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        pdl_kwargs = (
            {"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}
        )
        _pack_topk_ids_triton_kernel[grid](
            topk_ids,
            topk_weights,
            out,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
            **pdl_kwargs,
        )
        return out


@triton.jit
def _pack_topk_ids_triton_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    w = tl.load(topk_weights_ptr + offsets, mask=mask, other=0.0)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    w_bf16 = w.to(tl.bfloat16)
    w_i16 = w_bf16.to(tl.int16, bitcast=True)
    w_i32 = w_i16.to(tl.int32) & 0xFFFF

    ids_i32 = ids.to(tl.int32)
    packed = (ids_i32 << 16) | w_i32

    tl.store(out_ptr + offsets, packed, mask=mask)
