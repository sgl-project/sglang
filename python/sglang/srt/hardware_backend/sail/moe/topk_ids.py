"""The int32 topk ids SAIL's DeepEP dispatchers hand back.

``layers/moe/topk`` owns the padded-region masking; what is SAIL-specific is
which ids need widening to int64 at all, and whether the fused kernel can take
them. Both live here so the shared files keep no SAIL condition beyond the
one-line gate that imports this module.
"""

from typing import Optional

import torch

from sglang.srt.layers.moe.utils import is_deepep_class_backend


def need_int64_masking(topk_ids: torch.Tensor) -> bool:
    """Whether this id tensor has to be widened before the padded-region mask.

    A DeepEP-family dispatcher leaves the ids int32; the combine needs int64 for
    its index math, so the mask rewrites them directly in int64.
    """
    return topk_ids.dtype == torch.int32 and is_deepep_class_backend()


def mask_padded_region_to_int64(
    topk_ids: torch.Tensor,
    num_token_non_padded: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Widen int32 ids to int64 while masking the padded rows, or None when the
    fused kernel cannot take this tensor and the caller should fall back."""
    if not (
        topk_ids.is_cuda
        and topk_ids.dtype == torch.int32
        and topk_ids.shape[1] <= 32
        and topk_ids.is_contiguous()
    ):
        return None
    from sglang.kernels.ops.moe.dsv4 import mask_topk_ids_to_int64

    return mask_topk_ids_to_int64(topk_ids, num_token_non_padded)
