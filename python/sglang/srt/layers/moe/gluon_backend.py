from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


class GluonMoeBackend(ABC):
    """Strict whole-layer Gluon MoE backend.

    Gluon implementations may fuse routing, routed experts, and shared experts,
    so their boundary is intentionally above ``MoeRunnerCore``. SGLang still
    owns the post-expert collective. Implementations must either return a valid
    rank-local output or raise; falling back to another backend is forbidden.
    """

    @abstractmethod
    def bind(self, layer: torch.nn.Module, experts: FusedMoE) -> None:
        """Validate the layer and bind state needed before graph capture."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        gemm_output_zero_allocator=None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        skip_shared_experts: bool = False,
        num_token_non_padded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the backend or raise when the call contract is unsupported."""
