from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_align_module() -> Module:
    return load_jit(
        "moe_lora_align_block_size",
        cuda_files=["lora/moe_lora_align_kernel.cu"],
        cuda_wrappers=[
            ("moe_lora_align_block_size", "MoeLoraAlignBlockSizeKernel::run"),
        ],
    )


def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    seg_indptr: torch.Tensor,
    req_to_lora: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    maybe_expert_map: Optional[torch.Tensor] = None,
) -> None:
    module = _jit_moe_align_module()

    cumsum_buffer = torch.zeros(
        max_loras * (num_experts + 1), dtype=torch.int32, device=topk_ids.device
    )
    token_mask = torch.empty(
        (max_loras * topk_ids.shape[0],), dtype=torch.int32, device=topk_ids.device
    )

    module.moe_lora_align_block_size(
        topk_ids,
        seg_indptr,
        req_to_lora,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        maybe_expert_map,
        cumsum_buffer,
        token_mask,
    )
