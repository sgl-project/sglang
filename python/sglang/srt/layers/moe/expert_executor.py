from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


@dataclass(frozen=True)
class MoeExpertExecutorContext:
    """Model and runtime context for selecting an external expert executor."""

    model_family: str
    layer_id: int
    num_experts: int
    num_redundant_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    activation: str
    reduce_results: bool
    quant_config: Optional[QuantizationConfig]
    prefix: str
    tp_size: int
    moe_tp_size: int
    moe_ep_size: int
