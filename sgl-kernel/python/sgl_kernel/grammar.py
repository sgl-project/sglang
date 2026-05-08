from typing import List, Optional, Union

import torch


def apply_token_bitmask_inplace_cuda(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)
    if indices is not None:
        indices = indices.to(logits.device)
    torch.ops.sgl_kernel.apply_token_bitmask_inplace_cuda(logits, bitmask, indices)
