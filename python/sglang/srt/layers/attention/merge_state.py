from typing import Optional, Tuple

import torch

from sglang.srt.utils import is_cuda


def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    # NOTE(DefTruth): Currently, custom merge_state v2 CUDA kernel
    # is not support for FP8 dtype, fallback to use Triton kernel.
    def supported_dtypes(o: torch.Tensor) -> bool:
        return o.dtype in [torch.float32, torch.half, torch.bfloat16]

    # NOTE(DefTruth): Currently, custom merge_state v2 CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size (float32 -> 4, half/bfloat16 -> 8).
    def supported_headdim(o: torch.Tensor) -> bool:
        headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        if o.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0

    if is_cuda() and supported_dtypes(output) and supported_headdim(output):
        from sgl_kernel import merge_state_v2

        return merge_state_v2(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    else:
        from sglang.srt.layers.attention.triton_ops.merge_state import merge_state

        return merge_state(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
