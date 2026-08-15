# SPDX-License-Identifier: Apache-2.0

import torch


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Modulate by shift and scale."""
    if scale is None and shift is None:
        return x
    if shift is None:
        return x * (1 + scale.unsqueeze(1))  # type: ignore[union-attr]
    if scale is None:
        return x + shift.unsqueeze(1)  # type: ignore[union-attr]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
