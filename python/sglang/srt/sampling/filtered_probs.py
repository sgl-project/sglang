"""Apply the configured top-k/top-p order with backend renormalization ops."""

from typing import Callable, Optional

import torch


def renorm_top_k_top_p(
    probs: torch.Tensor,
    top_ks: Optional[torch.Tensor],
    top_ps: Optional[torch.Tensor],
    filter_apply_order: str,
    *,
    top_k_renorm: Callable,
    top_p_renorm: Callable,
) -> torch.Tensor:
    filters = ((top_k_renorm, top_ks), (top_p_renorm, top_ps))
    if filter_apply_order == "joint":
        # Top-p must measure mass on the full distribution before top-k.
        filters = filters[::-1]
    for renorm, threshold in filters:
        if threshold is not None:
            probs = renorm(probs, threshold)
    return probs
