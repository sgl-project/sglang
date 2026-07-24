"""Provider payloads for the SGL LoRA MoE pipeline.

Each payload describes both the model dimensions and the resident provider
representation.  Consumers must use ``num_local_experts``,
``intermediate_size``, and ``hidden_size`` for semantic sizing instead of
reverse-engineering dimensions from a resident tensor.

The tensors are borrowed from the base MoE layer and remain owned by that
layer; ``sgl_lora`` never copies or reorders weights on a forward.

BF16 only for now: FP8, native NVFP4 W4A4, and Marlin W4A16 payloads land
with their provider milestones (execution plan §2.2).
"""

from __future__ import annotations

import msgspec
import torch


class SglLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """Unquantized standard-layout MoE weights.

    ``w13_weight`` is ``[E_local, 2 * I, H]`` (gate first) and ``w2_weight``
    is ``[E_local, H, I]``.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
