from __future__ import annotations

import msgspec
import torch


class MoeLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """Unquantized standard-layout MoE weights.

    ``w13_weight`` is ``[E_local, S * I, H]`` with ``S=2`` when gated (gate
    first) and ``S=1`` when not; the pointwise activation is a separate axis
    and does not affect ``S``. ``w2_weight`` is ``[E_local, H, I]``.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
