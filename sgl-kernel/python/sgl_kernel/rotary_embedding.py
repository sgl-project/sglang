from typing import Optional
import torch

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
