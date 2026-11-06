"""Draft FFN modules built from DraftMixFFNSpec; dense now, MoE next."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch
from torch import nn

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix

SWIGLUOAI_ALPHA = 1.702  # gpt-oss's fixed sigmoid slope


def _silu_and_mul_clamp(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """DeepSeek-V4's SwiGLU: gate capped at limit, up clamped to [-limit, limit]."""
    from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp

    out = gate_up.new_empty(gate_up.shape[:-1] + (gate_up.shape[-1] // 2,))
    silu_and_mul_clamp(gate_up, out, limit)
    return out


def gated_activation(
    hidden_act: str,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The act(gate) * up of a draft FFN, from the spec's activation fields."""
    if hidden_act == "swigluoai":
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            swiglu_no_interleaved_with_alpha_and_limit,
        )

        if swiglu_limit is None:
            raise ValueError("Draft FFN: hidden_act='swigluoai' needs swiglu_limit.")
        return partial(
            swiglu_no_interleaved_with_alpha_and_limit,
            gemm1_alpha=float(swiglu_alpha or SWIGLUOAI_ALPHA),
            gemm1_limit=float(swiglu_limit),
        )
    if hidden_act != "silu":
        raise ValueError(
            f"Draft FFN: hidden_act={hidden_act!r} is not implemented; "
            "available are ('silu', 'swigluoai')."
        )
    if swiglu_limit is None:
        return SiluAndMul()
    return partial(_silu_and_mul_clamp, limit=float(swiglu_limit))


class DraftMLP(nn.Module):
    """Gated dense FFN of a draft layer; also the shared expert of an MoE layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        *,
        swiglu_limit: Optional[float] = None,
        swiglu_alpha: Optional[float] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = gated_activation(hidden_act, swiglu_limit, swiglu_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out
