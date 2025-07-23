from typing import Optional, Tuple

import torch
from sgl_kernel import merge_state_v2

from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


# Automatically fallback to the Triton kernel in some cases
# (e.g., for AMD GPUs, when the head dimension is not a multiple
# of 4 or 8, and in FP8 precision)
def _supported_dtypes(o: torch.Tensor) -> bool:
    return o.dtype in [torch.float32, torch.half, torch.bfloat16]


def _supported_headdim(o: torch.Tensor) -> bool:
    headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    if o.dtype == torch.float32:
        return headdim % 4 == 0
    return headdim % 8 == 0


def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if (
        _is_cuda
        and _supported_dtypes(prefix_output)
        and _supported_headdim(prefix_output)
    ):
        return merge_state_v2(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    else:
        # Fallback to Triton kernel
        return merge_state_triton(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
