"""Explicit input layout for a linear consuming prequantized MXFP8 activations."""

from typing import NamedTuple

import torch


class Mxfp8SwizzledInput(NamedTuple):
    """E4M3 activations and UE8M0 scales in FlashInfer's 128x4 layout.

    A plain FP8 tuple may contain block-FP8 scales with a different layout.
    This marker lets a converted block-FP8 linear distinguish the two.
    """

    data: torch.Tensor
    scales: torch.Tensor
