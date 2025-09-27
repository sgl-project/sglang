from typing import Optional

import torch


# Adapted from https://github.com/vllm-project/vllm/blob/9214e60631a79506e7669650de87806a123e0b0b/vllm/_custom_ops.py#L249
# pos encoding ops
def rotary_embedding(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    is_neox: bool,
) -> None:
    torch.ops.sgl_kernel.rotary_embedding.default(
        cos,
        sin,
        query,
        key,
        head_size,
        is_neox,
    )
