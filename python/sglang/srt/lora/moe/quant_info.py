from __future__ import annotations

import msgspec
import torch


class MoeLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """Unquantized MoE weights in the standard layout.

    ``w13_weight`` is ``[E_local, S * I, H]``. ``S`` is 2 when the layer
    gates, and the gate rows come first. ``S`` is 1 when the layer does not
    gate. The choice of pointwise function does not change ``S``.
    ``w2_weight`` is ``[E_local, H, I]``.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
