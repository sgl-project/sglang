import torch
from typing import Optional, List, Union, Tuple
from sglang.srt.layers.activation import GeluAndMul  # for variant 6


# =============================================================================
# 1. Standard SwiGLU
# =============================================================================
class NPUSwiglu:
    """
    Standard SwiGLU activation (NPU backend).
    """
    def _apply_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.npu.npu_swiglu(hidden_states)


# =============================================================================
# 2. Dequant‑SwiGLU‑Quant
# =============================================================================
class NPUSwigluQuant:
    """
    SwiGLU activation with built‑in dequant + re‑quant.
    Return (hidden_states, swiglu_out_scale).
    """
    def _apply_activation(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            quant_mode=1,
            activate_left=True,
        )
        return hidden_states, swiglu_out_scale


# =============================================================================
# 3. Dequant‑SwiGLU‑Quant with explicit weight/activation scales & bias
# =============================================================================
class NPUSwigluQuantWithScales:
    """
    SwiGLU activation with explicit per‑channel weight scale, per‑token
    activation scale, optional bias, and group info.
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
# 4. SwiGLU via sgl_kernel_npu custom swiglu_quant kernel for DeepEP scenarios
# =============================================================================
class NPUSwigluQuantKernel:
    """
    Uses the dedicated swiglu_quant kernel from sgl_kernel_npu.
    This kernel internally handles dequant, SwiGLU, and re‑quant.
    """
    def __init__(self):
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant
        self._kernel = swiglu_quant

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        need_quant: bool = True,  # If False, only dequant+swiglu (no re‑quant)
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._kernel(hidden_states, group_list, group_list_type, need_quant=need_quant)


# =============================================================================
# 5. GELU + Mul alternative (not SwiGLU, but used in non‑silu MoE)
# =============================================================================
class NPUGeluAndMul:
    """
    GELU + Mul activation (used instead of SwiGLU in some fused_moe_npu paths).
    """
    def _apply_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return GeluAndMul()(hidden_states)


# =============================================================================
# Unified dispatcher: picks the correct SwiGLU variant based on method
# =============================================================================
def get_swiglu_variant(
    method: str,
    **kwargs
) -> Union[
    NPUSwiglu,
    NPUSwigluQuant,
    NPUSwigluQuantWithScales,
    NPUSwigluQuantKernel,
    NPUGeluAndMul,
]:
    """
    Returns an appropriate activation object for the given NPU MoE method.

    Arguments:
        method: one of
            - "standard"
            - "dequant_swiglu_quant"
            - "dequant_swiglu_quant_with_scales"
            - "swiglu_quant_kernel"
            - "gelu_and_mul"
    """
    variants = {
        "standard": NPUSwiglu,
        "dequant_swiglu_quant": NPUSwigluQuant,
        "dequant_swiglu_quant_with_scales": NPUSwigluQuantWithScales,
        "swiglu_quant_kernel": NPUSwigluQuantKernel,
        "gelu_and_mul": NPUGeluAndMul,
    }
    if method not in variants:
        raise ValueError(f"Unknown SwiGLU variant: {method}")
    return variants[method](**kwargs)
