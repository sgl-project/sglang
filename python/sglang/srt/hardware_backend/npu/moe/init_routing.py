"""
NPU MoE init routing components.

Prepare token routing before expert computation. Two API versions are provided:
- v1: legacy routing using ``npu_moe_init_routing``.
- v2: improved routing using ``npu_moe_init_routing_v2``.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Tuple

import torch

# ``npu_moe_init_routing_v2`` quant_mode selecting MXFP8: the op emits an
# float8_e4m3fn payload plus an e8m0 block scale, fusing the activation quant
# that would otherwise need a separate ``npu_dynamic_mx_quant`` pass.
MXFP8_QUANT_MODE = 3
MXFP4_QUANT_MODE = 9
INT4_QUANT_MODE = 13


def _packed_payload(data, scale, source_shape, contract):
    from sglang.srt.hardware_backend.npu.moe.contracts import INT4, QuantizedOperand

    rows, hidden = source_shape
    if (
        contract == INT4
        and data.dtype == torch.uint8
        and tuple(data.shape) == (rows, hidden // 2)
    ):
        data = data.contiguous().view(torch.int32)
    if contract.block_size:
        scale = _normalize_mxfp_scale(scale).view(torch.uint8)
    else:
        scale = scale.reshape(rows)
    operand = QuantizedOperand(data, scale, contract, source_shape)
    operand.validate(contract)
    return data, scale


@lru_cache(maxsize=None)
def supports_packed_routing(kind, passthrough=False):
    """Check packed routing against the installed activation quantizer."""
    try:
        import torch_npu

        from sglang.srt.hardware_backend.npu.moe.contracts import INT4, MXFP4
        from sglang.srt.hardware_backend.npu.moe.quant import quantize_mxfp4_activation
        from sglang.srt.hardware_backend.npu.moe.tp_fusion import supports_npu_op
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            _get_float4_e2m1fn_x2_dtype,
        )

        if kind == "int4" and passthrough:
            return False
        if not supports_npu_op("npu_moe_init_routing_v2", ("x_dtype", "scale")):
            return False
        contract = MXFP4 if kind == "mxfp4" else INT4
        x = (
            torch.linspace(-2, 2, 256, device="npu", dtype=torch.float32)
            .reshape(2, 128)
            .bfloat16()
        )
        ids = torch.tensor([[0], [0]], device="npu", dtype=torch.int32)
        if kind == "mxfp4":
            expected, expected_scale = quantize_mxfp4_activation(
                x, _get_float4_e2m1fn_x2_dtype()
            )
            kwargs = {"quant_mode": MXFP4_QUANT_MODE}
            if passthrough:
                kwargs = {
                    "quant_mode": -1,
                    "x_dtype": torch_npu.float4_e2m1fn_x2,
                    "scale": expected_scale.view(torch.uint8),
                }
        else:
            expected, expected_scale = torch.ops.npu.npu_dynamic_quant(
                x, dst_type=torch.quint4x2
            )
            kwargs = {"quant_mode": INT4_QUANT_MODE}
        data, _, _, scale = torch.ops.npu.npu_moe_init_routing_v2(
            expected if passthrough else x,
            ids,
            expert_num=1,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, 1],
            **kwargs,
        )
        data, scale = _packed_payload(data, scale, tuple(x.shape), contract)
        expected, expected_scale = _packed_payload(
            expected, expected_scale, tuple(x.shape), contract
        )
        return torch.equal(data, expected) and torch.equal(
            scale.view(torch.uint8), expected_scale.view(torch.uint8)
        )
    except Exception:
        return False


def _normalize_mxfp_scale(scale: torch.Tensor) -> torch.Tensor:
    """Reshape a flat 2D e8m0 block scale ``[N, M]`` into pair-split ``[N, M//2, 2]``.

    ``npu_moe_init_routing_v2(quant_mode=3)`` emits the scale flat, while the
    grouped matmul wants the pair-split view. Already-3D scales (what
    ``npu_dynamic_mx_quant`` returns) pass through untouched.
    """
    if scale is None or scale.ndim != 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


class BaseInitRouting(ABC):
    """Abstract base for NPU MoE init routing."""

    @abstractmethod
    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: ...


class NPUMoEInitRouting_v1(BaseInitRouting):
    """
    NPU MoE init routing (v1 API).

    Uses ``npu_moe_init_routing`` with a manually constructed ``row_idx`` tensor.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_ids.device)
            .view(topk_ids.shape[1], -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            )
        )
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, None


class NPUMoEInitRouting_v2(BaseInitRouting):
    """
    NPU MoE init routing (v2 API).

    Uses ``npu_moe_init_routing_v2``, which integrates expert token counting.
    """

    def __init__(self, quant_mode: int = -1):
        self.quant_mode = quant_mode

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
        input_scale: Optional[torch.Tensor] = None,
        source_shape=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        logical_width = (
            hidden_states.shape[1] if source_shape is None else source_shape[1]
        )
        quant_mode = self.quant_mode if input_scale is None else -1
        scale_kwargs = {}
        if input_scale is not None:
            if self.quant_mode in (MXFP8_QUANT_MODE, MXFP4_QUANT_MODE):
                scale_kwargs["scale"] = input_scale.view(torch.uint8)
                if self.quant_mode == MXFP4_QUANT_MODE:
                    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
                        _get_float4_e2m1fn_x2_dtype,
                    )

                    if not supports_packed_routing("mxfp4", True):
                        raise ValueError("This runtime cannot route prequantized MXFP4")
                    scale_kwargs["x_dtype"] = _get_float4_e2m1fn_x2_dtype()
            elif self.quant_mode == 1 and hidden_states.dtype == torch.int8:
                scale_kwargs["scale"] = input_scale.flatten()
            else:
                raise ValueError(
                    "Prequantized routing requires matching INT8/MXFP8 input"
                )
        hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=quant_mode,
                **scale_kwargs,
            )
        )
        if input_scale is not None:
            pertoken_scale = pertoken_scale.view(input_scale.dtype)
        elif self.quant_mode == -1:
            pertoken_scale = None
        elif self.quant_mode == MXFP8_QUANT_MODE:
            pertoken_scale = _normalize_mxfp_scale(pertoken_scale)
        if self.quant_mode in (MXFP4_QUANT_MODE, INT4_QUANT_MODE):
            from sglang.srt.hardware_backend.npu.moe.contracts import INT4, MXFP4

            contract = MXFP4 if self.quant_mode == MXFP4_QUANT_MODE else INT4
            hidden_states, pertoken_scale = _packed_payload(
                hidden_states,
                pertoken_scale,
                (num_tokens * top_k, logical_width),
                contract,
            )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale


class NPUMoEInitRouting_Quant(BaseInitRouting):
    """
    NPU MoE init routing (Quant API).

    Uses ``npu_moe_init_routing_quant``, which integrates expert token counting.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]

        hidden_states, expanded_row_idx, expert_tokens, _, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_quant(
                hidden_states,
                topk_ids,
                active_num=num_tokens * topk_ids.shape[1],
                expert_num=num_experts,
                expert_tokens_num_mode=1,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale
