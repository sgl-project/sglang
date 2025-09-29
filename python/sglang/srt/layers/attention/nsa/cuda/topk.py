from __future__ import annotations

from typing import Any

import torch

from .utils import load_kernel_module


def _load_topk_module() -> Any:
    """
    Load the index manipulation module.
    """
    return load_kernel_module("topk.cu", "topk_kernel")


# TODO(dark): configure out why my cuda impl is a little slower....
# I believe it has something to do with unrolling loops (?)
_USE_TL = True


def fast_topk(
    score: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    return _load_topk_module().fast_topk(score, indices, lengths, _USE_TL)


def fast_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return _load_topk_module().fast_topk_transform(
        score, lengths, dst_page_table, src_page_table, cu_seqlens, _USE_TL
    )
