"""
Hidden state quantization utilities for NPU MoE.

Each class quantises hidden states and returns a (quantized_tensor, scale) tuple.
For static quantization the scale is ``None``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseHiddenStatesQuant(ABC):
    """Abstract base for NPU hidden state quantisation."""

    def __init__(self, quant_dtype: torch.dtype) -> None:
        self.quant_dtype = quant_dtype

    @abstractmethod
    def __call__(
        self, hidden_states: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...


class HiddenStatesDynamicQuant(BaseHiddenStatesQuant):
    """
    Dynamic per‑token quantisation of hidden states.

    Returns ``(quantized_hidden_states, per‑token_scale)``.
    """

    def __call__(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, scale = torch.ops.npu.npu_dynamic_quant(
            hidden_states, dst_type=self.quant_dtype
        )
        return quantized, scale


class HiddenStatesStaticQuant(BaseHiddenStatesQuant):
    """
    Static quantisation using pre‑computed layer‑specific scales and offsets.

    The ``layer`` argument must expose ``aclnn_input_scale_reciprocal`` and
    ``aclnn_input_offset``. Returns ``(quantized_hidden_states, None)``.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        layer: torch.nn.Module,
    ) -> Tuple[torch.Tensor, None]:
        if not hasattr(layer, "aclnn_input_scale_reciprocal") or not hasattr(
            layer, "aclnn_input_offset"
        ):
            raise AttributeError(
                "Static quantisation requires layer attributes "
                "'aclnn_input_scale_reciprocal' and 'aclnn_input_offset'."
            )

        quantized = torch.ops.npu.npu_quantize(
            hidden_states,
            layer.aclnn_input_scale_reciprocal,
            layer.aclnn_input_offset,
            self.quant_dtype,
            -1,
            False,
        )
        return quantized, None


class HiddenStatesMXFP8DynamicQuant(BaseHiddenStatesQuant):
    """
    Dynamic MXFP8 per‑token quantisation for NPU MoE hidden states.

    Uses the NPU dynamic MX quantisation API to produce
    ``float8_e4m3fn`` quantized activations and
    ``float8_e8m0fnu`` per‑token scales.
    """

    def __init__(self) -> None:
        # quant_dtype is not needed here because npu_dynamic_mx_quant
        # will always output float8_e4m3fn
        super().__init__(torch.float8_e4m3fn)

    def __call__(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, scale = torch.ops.npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn
        )
        return quantized, scale
