"""Mixture-of-Experts routing / bookkeeping kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_CUDA = CapabilityRequirement(requires_cuda=True)

register_kernel(
    KernelSpec(
        op="moe.moe_align_block_size",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:moe_align_block_size",
        format_signature=FormatSignature(
            in_place=True,
            description="align/sort expert token ids into block-padded buffers",
        ),
        description="MoE align-block-size (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="moe.moe_align_block_size",
        backend=KernelBackend.CUDA_JIT,
        target="sglang.jit_kernel.moe_align:moe_align_block_size",
        capability=_CUDA,
        format_signature=FormatSignature(
            in_place=True,
            description="MoE align-block-size (JIT variant, AOT signature)",
        ),
        description="MoE align-block-size (sglang.jit_kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="moe.topk_softmax",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:topk_softmax",
        format_signature=FormatSignature(
            in_place=True,
            description="top-k softmax routing weights/ids",
        ),
        description="MoE top-k softmax (sgl_kernel wheel).",
    )
)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
    ignore_invalid_expert: bool = False,
) -> None:
    """Align and sort expert token ids into block-padded output buffers."""
    kernel = get_kernel("moe.moe_align_block_size", KernelBackend.CUDA_AOT)
    if ignore_invalid_expert:
        return kernel(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            pad_sorted_token_ids,
            ignore_invalid_expert,
        )
    return kernel(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """Compute top-k softmax routing weights/ids for MoE."""
    # The XPU sgl-kernel build (sgl-kernel-xpu) exposes
    # topk_softmax(topk_weights, topk_ids, gating_output, renormalize=False) and
    # does not accept moe_softcapping / correction_bias. Forward only the args the
    # kernel supports; guard so a caller that actually needs the extras fails loudly
    # instead of having them silently dropped.
    assert (
        moe_softcapping == 0.0 and correction_bias is None
    ), "moe_softcapping / correction_bias are not supported by the XPU topk_softmax kernel"
    return get_kernel("moe.topk_softmax", KernelBackend.CUDA_AOT)(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
    )


__all__ = ["moe_align_block_size", "topk_softmax"]


# Fused MoE-LoRA Triton kernels migrated into this group (from lora/triton_ops);
# registered for inventory. Import them from their modules.
_TRITON_KERNELS = [
    ("fused_moe_lora_kernel", "fused_moe_lora"),
    ("virtual_experts", "merged_experts_fused_moe_lora_add"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"moe.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.moe.{_mod}:{_fn}",
        )
    )
del _mod, _fn


# Triton kernels migrated from srt/layers/moe (RFC #29630, Phase 2.5);
# registered for inventory. Import them from their modules.
_PHASE25_TRITON_KERNELS = [
    ("ep_moe_kernels", "deepep_run_moe_deep_preprocess"),
    ("ep_moe_kernels", "deepep_permute_triton_kernel"),
    ("ep_moe_kernels", "deepep_post_reorder_triton_kernel"),
    ("fused_moe_triton_kernels", "invoke_fused_moe_kernel"),
    ("fused_moe_triton_kernels", "fused_moe_kernel"),
    ("fused_moe_triton_kernels", "fused_moe_kernel_gptq_awq"),
    ("mxfp8_moe_amd_gfx95", "fused_experts_mxfp8"),
    ("rocm_moe_utils", "upscale"),
    ("rocm_moe_utils", "upscale_mxfp4"),
    ("router", "fused_moe_router_shim"),
    ("deepep_waterfill_kernels", "materialize_waterfill_dispatch_fused"),
    ("fill_padded_rows", "_fill_padded_rows"),
]
for _mod, _fn in _PHASE25_TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"moe.{_fn.lstrip('_')}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.moe.{_mod}:{_fn}",
        )
    )
del _mod, _fn

# Packed (topk_id << 16 | bf16-weight) kernel migrated from
# srt/layers/quantization/mxfp4_flashinfer_trtllm_moe (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="moe.pack_topk_ids",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.moe.pack_topk_ids:PackTopkIds.triton",
    )
)
