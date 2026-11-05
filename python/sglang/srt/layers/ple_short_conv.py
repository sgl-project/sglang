"""Bound the temporary output of the PLE prefill depthwise convolution."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def chunked_packed_short_conv(
    conv_input: torch.Tensor,
    weight: torch.Tensor,
    lengths: Sequence[int],
    *,
    dilation: int,
    chunk_size: int,
) -> torch.Tensor:
    """Convolve ``[history | padded tokens]`` into packed request-order rows.

    Each output interval [start, end) reads the identical input interval
    [start, end + (kernel_size - 1) * dilation). This preserves causal history
    across chunk boundaries without updating state between chunks. Keep the
    same native conv1d and leave SiLU and all state/track commits to the caller.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    halo = (weight.shape[2] - 1) * dilation
    output = conv_input.new_empty((sum(lengths), conv_input.shape[1]))
    cursor = 0
    for request, length in enumerate(lengths):
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            # Match the stock contiguous input while bounding any layout copy.
            slab = conv_input[request : request + 1, :, start : end + halo].contiguous()
            chunk = F.conv1d(
                slab, weight, bias=None, dilation=dilation, groups=conv_input.shape[1]
            )
            output[cursor + start : cursor + end].copy_(chunk[0].transpose(0, 1))
            # Do not retain the previous output while allocating the next one.
            del chunk, slab
        cursor += length
    return output
