"""
Quantization kernel wrappers for NPU MoE.

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

    ``torch.float8_e4m3fn`` selects the MX (block-scaled) op, whose scale is a
    ``float8_e8m0fnu`` block scale ``[N, K//64, 2]`` rather than one scalar per
    token; the int8/int4 dtypes keep the plain per-token op.

    Returns ``(quantized_hidden_states, per‑token_scale)``.
    """

    def __init__(self, quant_dtype: torch.dtype) -> None:
        super().__init__(quant_dtype)
        if quant_dtype == torch.float8_e4m3fn:
            self._op = torch.ops.npu.npu_dynamic_mx_quant
        elif quant_dtype in (torch.int8, torch.quint4x2):
            self._op = torch.ops.npu.npu_dynamic_quant
        else:
            raise ValueError(f"Unsupported dynamic quant dtype: {quant_dtype}")

    def __call__(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, scale = self._op(hidden_states, dst_type=self.quant_dtype)
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
        # Optional defensive check (as suggested in the review)
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
