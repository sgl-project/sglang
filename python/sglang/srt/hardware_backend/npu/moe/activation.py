from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.activation import GeluAndMul


# =============================================================================
# Abstract base for all activation variants
# =============================================================================
class BaseActivation(ABC):
    """
    Abstract interface for NPU MoE activation components.

    Every subclass must implement ``_apply_activation`` and return a tuple
    ``(hidden_states, scale)`` where ``scale`` may be ``None`` if no
    quantisation scale is produced.
    """

    @abstractmethod
    def _apply_activation(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...


# =============================================================================
# 1. Standard SwiGLU
# =============================================================================
class NPUSwiglu(BaseActivation):
    """
    Standard SwiGLU activation (NPU backend).

    Use when:
        - Running without activation quantisation (e.g. W8A8 decode).
        - The downstream matmul does not need an activation scale.
    """

    def _apply_activation(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return torch.ops.npu.npu_swiglu(hidden_states), None


# =============================================================================
# 2. Dequant‑SwiGLU‑Quant
# =============================================================================
class NPUSwigluQuant(BaseActivation):
    """
    SwiGLU activation with built‑in dequant + re‑quant.

    Use when:
        - The layer is quantised (weight + activation) and you need a
          fused kernel that dequantises, applies SwiGLU, and re‑quantises
          the hidden states for the next operation.
        - You are running standard W8A8/W4A4 MoE inference.
    """

    def _apply_activation(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            quant_mode=1,
            activate_left=True,
        )
        return hidden_states, swiglu_out_scale


# =============================================================================
# 3. Dequant‑SwiGLU‑Quant with explicit scales & bias
# =============================================================================
class NPUSwigluQuantWithScales(BaseActivation):
    """
    SwiGLU activation with explicit per‑channel weight scale, per‑token
    activation scale, optional bias, and group info.

    Use when:
        - The quantisation scheme provides separate weight and activation scales,
          possibly with a bias, quant scale, and group index (e.g. advanced
          W4A8 or custom per‑group quantisation).
        - You need fine‑grained control over the quantisation parameters
          inside the fused kernel.
    """

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        group_index: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        quant_scale: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            group_index=group_index,
            activate_left=True,
            quant_mode=1,
        )
        return hidden_states, swiglu_out_scale


# =============================================================================
# 4. SwiGLU via sgl_kernel_npu custom swiglu kernel (DeepEP scenarios)
# =============================================================================
class NPUSwigluDeepEPKernel(BaseActivation):
    """
    Uses the dedicated ``swiglu_quant`` kernel from ``sgl_kernel_npu``.

    Use when:
        - Running DeepEP scenarios where the kernel supports dynamic
          dequant, SwiGLU, and optional re‑quant in a single pass.
        - You need the flexibility to turn off re‑quantisation by
          setting ``need_quant=False`` and still obtain a clean tuple output.
    """

    def __init__(self, need_quant: bool = True) -> None:
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

        self._kernel = swiglu_quant
        self.need_quant = need_quant

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        result = self._kernel(
            hidden_states, group_list, group_list_type, need_quant=self.need_quant
        )
        if isinstance(result, tuple):
            return result
        # If need_quant=False the kernel returns plain hidden_states (no scale)
        return result, None


# =============================================================================
# 5. GELU + Mul alternative
# =============================================================================
class NPUGeluAndMul(BaseActivation):
    """
    GELU + Mul activation (used instead of SwiGLU in some fused_moe_npu paths).

    Use when:
        - The model uses a GELU‑based activation instead of SwiGLU
          (e.g. certain non‑silu MoE variants).
        - The downstream code expects the same ``(hidden_states, scale)`` interface.
    """

    def __init__(self):
        self._gelu = GeluAndMul()

    def _apply_activation(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._gelu(hidden_states), None


# =============================================================================
# 6. SwiGLU OAI (OpenAI‑style fused kernel)
# =============================================================================
class NPUSwigluOAI(BaseActivation):
    """
    SwiGLU activation using the ``sgl_kernel_npu`` ``swiglu_oai`` kernel.

    Use when:
        - The model configuration specifies ``activation="npu_swiglu_oai"``.
        - You need the fused, highly‑optimised kernel that may take
          additional layer metadata.
    """

    def __init__(self, layer: Any) -> None:
        from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

        self._kernel = swiglu_oai
        self._layer = layer

    def _apply_activation(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._kernel(self._layer, hidden_states), None


# =============================================================================
# 7. SwiGLU with step‑and‑mul clamping
# =============================================================================
class NPUSwigluStepAndMul(BaseActivation):
    """
    SwiGLU activation with optional clamp limit (step‑and‑mul style).

    Use when:
        - The model uses ``activation="silu"`` and a ``gemm1_clamp_limit``
          is provided in the layer configuration.
        - Falls back to the standard ``npu_swiglu`` when no clamp limit
          is set.
    """

    def __init__(self, clamp_limit: Optional[float] = None) -> None:
        self._clamp_limit = clamp_limit

    def _apply_activation(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._clamp_limit is not None:
            # Use the pure PyTorch implementation of clamped SwiGLU
            return self.swiglustep_and_mul(hidden_states, self._clamp_limit), None
        # Fallback to standard SiLU without clamping
        return torch.ops.npu.npu_swiglu(hidden_states), None

    @staticmethod
    def swiglustep_and_mul(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
        """Out‑of‑place SwiGLU step activation.

        Splits the input tensor into two halves, applies SiLU to the first half
        and clamps it to ``[0, limit]``, clamps the second half to
        ``[-limit, limit]``, then multiplies them element‑wise.

        Args:
            x: Input tensor of shape ``(..., 2 * d)``.
            limit: Clamp value (default 7.0).

        Returns:
            Activated tensor of shape ``(..., d)``.
        """
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate).clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * up


# =============================================================================
# Unified dispatcher
# =============================================================================
def get_swiglu_variant(method: str, **kwargs: Any) -> BaseActivation:
    """
    Returns an appropriate activation object for the given NPU MoE method.

    Args:
        method: one of
            - ``"standard"``
            - ``"dequant_swiglu_quant"``
            - ``"dequant_swiglu_quant_with_scales"``
            - ``"swiglu_quant_deepep_kernel"``
            - ``"gelu_and_mul"``
            - ``"swiglu_oai"`` (requires ``layer`` kwarg)
            - ``"swiglustep_and_mul"`` (accepts optional ``clamp_limit``)
        **kwargs: Additional constructor arguments for parameterised variants.

    Returns:
        An instance of a ``BaseActivation`` subclass.

    Raises:
        ValueError: if ``method`` is not recognised or required kwargs are missing.
    """
    variants: dict[str, type[BaseActivation]] = {
        "standard": NPUSwiglu,
        "dequant_swiglu_quant": NPUSwigluQuant,
        "dequant_swiglu_quant_with_scales": NPUSwigluQuantWithScales,
        "swiglu_quant_deepep_kernel": NPUSwigluDeepEPKernel,  # corrected name
        "gelu_and_mul": NPUGeluAndMul,
    }

    if method == "swiglu_oai":
        layer = kwargs.pop("layer", None)
        if layer is None:
            raise ValueError("layer is required for swiglu_oai activation")
        return NPUSwigluOAI(layer)

    if method == "swiglustep_and_mul":
        clamp_limit = kwargs.pop("clamp_limit", None)
        return NPUSwigluStepAndMul(clamp_limit=clamp_limit)

    if method not in variants:
        raise ValueError(f"Unknown SwiGLU variant: {method}")

    return variants[method]()
