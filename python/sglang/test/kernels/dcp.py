"""Pure-torch reference for the DCP LSE-weighted combine kernel."""

import torch


def lse_weighted_combine_cpu(
    partial_outputs: torch.Tensor,
    partial_lses: torch.Tensor,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """CPU reference: combine N partial attention outputs using LSE weights.

    Args:
        partial_outputs: [N, B, H_local, D]
        partial_lses:    [N, B, H_local]
        is_lse_base_on_e: base-e (True) or base-2 (False)

    Returns:
        [B, H_local, D] combined output
    """
    N, B, H_local, D = partial_outputs.shape
    partial_outputs = partial_outputs.float()
    partial_lses = partial_lses.float()

    # Sanitize
    partial_lses = torch.where(
        torch.isnan(partial_lses) | torch.isinf(partial_lses),
        torch.full_like(partial_lses, float("-inf")),
        partial_lses,
    )

    # max LSE for numerical stability
    lse_max, _ = partial_lses.max(dim=0)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    centered = partial_lses - lse_max.unsqueeze(0)
    if is_lse_base_on_e:
        weights = torch.exp(centered)
    else:
        weights = torch.pow(2.0, centered)

    weight_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weight_sum.clamp_min(1.0)

    combined = (partial_outputs * weights.unsqueeze(-1)).sum(dim=0)
    return combined
