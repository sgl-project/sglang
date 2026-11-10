# SPDX-License-Identifier: Apache-2.0
"""Weight folds shared by the VAE decoder fast paths."""

from __future__ import annotations

import torch
from torch import nn

# Output phase a (0..3) of a ConvTranspose2d(k4, s2, p1) equals the sum of these
# 3x3 taps of the original conv applied after a nearest 2x upsample.
UPSAMPLE_TAP_MAP = {0: (2,), 1: (1, 2), 2: (0, 1), 3: (0,)}


def fold_upsample2x_conv2d_weight(conv: nn.Conv2d) -> torch.Tensor:
    """Sum the 3x3 conv taps into the ConvTranspose2d(k4) kernel equal to nearest-2x + conv3x3(p1).

    The sum runs in fp32 and is rounded once to the conv's dtype, so the fold
    differs from the eager chain at rounding level. Keeps the weight's
    channels_last layout when it has one.
    """
    if not (
        type(conv) is nn.Conv2d
        and conv.kernel_size == (3, 3)
        and conv.stride == (1, 1)
        and conv.padding == (1, 1)
        and conv.dilation == (1, 1)
        and conv.groups == 1
        and conv.padding_mode == "zeros"
    ):
        raise ValueError(
            "fold_upsample2x_conv2d_weight needs a plain Conv2d(k3, s1, p1, zeros), got "
            f"{type(conv).__name__} kernel={getattr(conv, 'kernel_size', None)} "
            f"stride={getattr(conv, 'stride', None)} padding={getattr(conv, 'padding', None)} "
            f"dilation={getattr(conv, 'dilation', None)} groups={getattr(conv, 'groups', None)} "
            f"padding_mode={getattr(conv, 'padding_mode', None)}"
        )
    w = conv.weight.detach().float()  # [Cout, Cin, 3, 3]
    cout, cin = w.shape[:2]
    wt = w.new_zeros(cin, cout, 4, 4)  # ConvTranspose2d layout
    for a in range(4):
        for b in range(4):
            acc = w.new_zeros(cout, cin)
            for i in UPSAMPLE_TAP_MAP[a]:
                for j in UPSAMPLE_TAP_MAP[b]:
                    acc += w[:, :, i, j]
            wt[:, :, a, b] = acc.t()
    wt = wt.to(conv.weight.dtype)
    if conv.weight.is_contiguous(memory_format=torch.channels_last):
        wt = wt.contiguous(memory_format=torch.channels_last)
    return wt.to(conv.weight.device)
