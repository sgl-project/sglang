"""Mixture-of-Experts routing / bookkeeping kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

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

_CUDA = frozenset({CapabilityRequirement.CUDA})

register_kernel(
    KernelSpec(
        op="moe.moe_align_block_size_out",
        backend=KernelBackend.AOT,
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
        op="moe.moe_align_block_size_out",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.moe_align:moe_align_block_size_out",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True,
            description="MoE align-block-size (JIT variant, AOT signature)",
        ),
        description="MoE align-block-size (sglang.kernels.jit).",
    )
)
register_kernel(
    KernelSpec(
        op="moe.topk_softmax",
        backend=KernelBackend.AOT,
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
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    # Imported here, not in the module body: the dispatch module pulls in torch
    # and resolves a backend at import, and importing a group package has to stay
    # metadata-only (see `sglang.kernels.ops`).
    from sglang.kernels.ops.moe.moe_align_dispatch import align_block_size

    return align_block_size(topk_ids, block_size, num_experts, ignore_invalid_expert)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """Compute top-k softmax routing weights/ids for MoE."""
    return get_kernel("moe.topk_softmax", KernelBackend.AOT)(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        moe_softcapping,
        correction_bias,
    )


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
    ("moe_fused_mul_sum", "moe_fused_mul_sum"),
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

__all__ = ["moe_align_block_size", "topk_softmax"]
