"""Context projection for K3 pipeline-parallel DSpark prefill."""

from bisect import bisect_left
from typing import Optional

import torch
import torch.nn.functional as F


def context_feature_slice(
    layer_ids: list[int], start_layer: int, end_layer: int, is_last_rank: bool
) -> slice:
    if not layer_ids or layer_ids != sorted(set(layer_ids)):
        raise ValueError("DSpark PP requires sorted, unique target layer ids.")
    # K3 captures the mixture computed by the NEXT consumer. The first layer
    # on this stage owns the preceding stage's boundary capture.
    return slice(
        bisect_left(layer_ids, max(0, start_layer - 1)),
        bisect_left(layer_ids, end_layer if is_last_rank else end_layer - 1),
    )


def accumulate_context(
    hidden: Optional[torch.Tensor],
    accumulated: Optional[torch.Tensor],
    weight: torch.Tensor,
    features: slice,
    num_tokens: int,
) -> Optional[torch.Tensor]:
    """Sum FC column-block products; normalize only after the final stage.

    The wire carries [tokens, hidden_size], never the concatenated captures.
    Accumulate in FP32 to avoid rounding the running sum at every PP hop.
    """
    hidden_size = weight.shape[0]
    if features.start > 0 and accumulated is None:
        raise RuntimeError("Missing DSpark context from the preceding PP stage.")
    if accumulated is not None and accumulated.shape != (num_tokens, hidden_size):
        raise RuntimeError("DSpark PP context has an unexpected token/feature shape.")
    if features.start == features.stop:
        return accumulated
    width = (features.stop - features.start) * hidden_size
    if hidden is None or hidden.shape[0] < num_tokens or hidden.shape[1] != width:
        raise RuntimeError("Missing or incorrectly shaped local DSpark PP captures.")
    partial = F.linear(
        hidden[:num_tokens],
        weight[:, features.start * hidden_size : features.stop * hidden_size],
    ).float()
    return partial if accumulated is None else accumulated + partial
