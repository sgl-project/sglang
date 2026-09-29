from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


_BACKEND_ATTR = "_gluon_moe_backend"


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
    def prepare_weights(self) -> None:
        """Materialize the kernel layout after checkpoint weights are loaded."""

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


def bind_gluon_moe_backend(layer: torch.nn.Module, backend: GluonMoeBackend) -> None:
    """Attach a strict whole-layer backend to a model MoE and its experts."""

    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    if not get_moe_runner_backend().is_gluon():
        raise RuntimeError(
            "A Gluon MoE backend can only be bound when "
            "--moe-runner-backend gluon is selected"
        )
    if not isinstance(backend, GluonMoeBackend):
        raise TypeError("backend must implement GluonMoeBackend")
    if getattr(layer, _BACKEND_ATTR, None) is not None:
        raise RuntimeError("A Gluon MoE backend is already bound")

    experts = layer.experts
    backend.bind(layer, experts)
    setattr(layer, _BACKEND_ATTR, backend)
    # Quantization post-processing is invoked on the FusedMoE child. Keeping
    # the same adapter there lets it prepare the whole-layer layout after the
    # checkpoint has been loaded, rather than lazily in the first forward.
    setattr(experts, _BACKEND_ATTR, backend)


def prepare_gluon_moe_weights(experts: torch.nn.Module) -> None:
    """Prepare an attached backend, if any, during quant post-processing."""

    backend = getattr(experts, _BACKEND_ATTR, None)
    if backend is not None:
        backend.prepare_weights()


def should_use_gluon_moe(layer: torch.nn.Module) -> bool:
    """Return whether Gluon owns this layer, raising instead of falling back."""

    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    if not get_moe_runner_backend().is_gluon():
        return False
    if getattr(layer, _BACKEND_ATTR, None) is None:
        raise RuntimeError(
            "--moe-runner-backend gluon was selected, but no Gluon "
            f"implementation was bound to MoE layer {layer.layer_id}"
        )
    return True


def forward_gluon_moe(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Run and validate the output contract of an attached Gluon backend."""

    if not should_use_gluon_moe(layer):
        raise RuntimeError("Gluon MoE forward called without selecting Gluon")
    output = getattr(layer, _BACKEND_ATTR).forward(hidden_states, **kwargs)
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != hidden_states.shape
        or output.dtype != hidden_states.dtype
        or output.device != hidden_states.device
    ):
        raise RuntimeError("Gluon MoE output must match the input tensor contract")
    return output
