from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.runtime_context import get_parallel


# =============================================================================
# Abstract base for all activation variants
# =============================================================================
class BaseActivation(ABC):
    @abstractmethod
    def _apply_activation(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...


# =============================================================================
# Concrete activation implementations (unchanged except removed 8.)
# =============================================================================
class NPUSwiglu(BaseActivation):
    def _apply_activation(self, hidden_states: torch.Tensor):
        return torch.ops.npu.npu_swiglu(hidden_states), None


class NPUSwigluQuant(BaseActivation):
    def _apply_activation(self, hidden_states: torch.Tensor):
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            quant_mode=1,
            activate_left=True,
        )
        return hidden_states, swiglu_out_scale


class NPUSwigluQuantWithScales(BaseActivation):
    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        group_index: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        quant_scale: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
    ):
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


class NPUSwigluDeepEPKernel(BaseActivation):
    def __init__(self, need_quant: bool = True):
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

        self._kernel = swiglu_quant
        self.need_quant = need_quant

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        hidden_states, per_token_scale = self._kernel(
            hidden_states, group_list, group_list_type, need_quant=self.need_quant
        )
        if self.need_quant:
            return hidden_states, per_token_scale
        return hidden_states, None


class NPUGeluAndMul(BaseActivation):
    def __init__(self):
        self._gelu = GeluAndMul()

    def _apply_activation(self, hidden_states: torch.Tensor):
        return self._gelu(hidden_states), None


class NPUSwigluOAI(BaseActivation):
    def __init__(self, moe_runner_config=None):
        from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai_triton

        self._kernel = swiglu_oai_triton
        self._moe_runner_config = moe_runner_config

    def _apply_activation(self, hidden_states: torch.Tensor):
        # hidden_states is the output of the grouped matmul with shape
        # [num_tokens, 2 * inter].  The old swiglu_oai kernel derived the
        # gate_up dimension from layer.w13_weight.shape[2], which now fails
        # because w13_weight is stored un-transposed.  Instead we pass
        # the gate_up dimension explicitly from the tensor itself.
        alpha = 1.0
        clamp = None
        if self._moe_runner_config is not None:
            alpha = getattr(self._moe_runner_config, "gemm1_alpha", 1.0)
            clamp = getattr(self._moe_runner_config, "gemm1_clamp_limit", None)

        output = self._kernel(
            hidden_states,
            hidden_states.shape[-1],  # gate_up dim = 2 * inter
            alpha,
            clamp,
        )
        return output, None


class NPUSwigluStepAndMul(BaseActivation):
    def __init__(self, clamp_limit: Optional[float] = None):
        self._clamp_limit = clamp_limit

    def _apply_activation(self, hidden_states: torch.Tensor):
        if self._clamp_limit is not None:
            return self._swiglustep_and_mul(hidden_states, self._clamp_limit), None
        return torch.ops.npu.npu_swiglu(hidden_states), None

    @staticmethod
    def _swiglustep_and_mul(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate).clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * up


# =============================================================================
# Generic TP all‑gather wrapper – used by the runner when needed
# =============================================================================
class AllGatherActivationWrapper(BaseActivation):
    """
    Wraps any activation and adds an all‑gather along `dim` if TP > 1.

    This allows the runner to stay TP‑agnostic: the wrapper is applied
    transparently at construction time.
    """

    def __init__(self, inner: BaseActivation, dim: int = -1):
        self.inner = inner
        self.dim = dim

    def _apply_activation(self, *args, **kwargs):
        out, scale = self.inner._apply_activation(*args, **kwargs)
        if get_parallel().tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=self.dim)
        return out, scale


# =============================================================================
# Factory (unchanged, returns *base* activations)
# =============================================================================
def get_swiglu_variant(method: str, **kwargs: Any) -> BaseActivation:
    variants: dict[str, type[BaseActivation]] = {
        "standard": NPUSwiglu,
        "dequant_swiglu_quant": NPUSwigluQuant,
        "dequant_swiglu_quant_with_scales": NPUSwigluQuantWithScales,
        "swiglu_quant_deepep_kernel": NPUSwigluDeepEPKernel,
        "gelu_and_mul": NPUGeluAndMul,
    }
    if method == "swiglu_oai":
        # The OAI variant now uses the triton kernel that derives the gate_up
        # dimension from the tensor itself.  No extra parameters are needed.
        return NPUSwigluOAI()
    if method == "swiglustep_and_mul":
        clamp_limit = kwargs.pop("clamp_limit", None)
        return NPUSwigluStepAndMul(clamp_limit=clamp_limit)
    if method not in variants:
        raise ValueError(f"Unknown SwiGLU variant: {method}")
    return variants[method]()
