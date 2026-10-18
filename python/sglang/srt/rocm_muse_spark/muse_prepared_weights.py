"""Lossless permutation of native Muse FP4 words into MFMA load order.

No values or scales change. The last two four-element axes select the MFMA
K-step and lane-row, respectively; swapping them removes runtime lane swaps.
"""

import torch


def prepare(weight):
    assert weight.dtype == torch.uint8 and weight.is_contiguous()
    e, n, k2 = weight.shape
    assert n % 16 == 0 and k2 % 64 == 0
    words = weight.view(torch.int32)
    return (
        words.reshape(e, n // 16, 16, k2 // 64, 4, 4)
        .permute(0, 1, 3, 5, 2, 4)
        .contiguous()
        .view(torch.uint8)
        .reshape(e, n, k2)
    )


def restore(weight):
    e, n, k2 = weight.shape
    return (
        weight.view(torch.int32)
        .reshape(e, n // 16, k2 // 64, 4, 16, 4)
        .permute(0, 1, 4, 2, 5, 3)
        .contiguous()
        .view(torch.uint8)
        .reshape(e, n, k2)
    )
