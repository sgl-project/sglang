"""Standalone Double Sparsity selection for DeepSeek-V3.2 (FP8).

Hooks into ``DeepseekV2AttentionMLA.forward_core`` via a single config-gated
branch. Does NOT register with the HiSparse algorithm registry, does NOT
require ``--enable-hisparse``, and does NOT require PD-disaggregation.
"""

from sglang.srt.layers.attention.double_sparsity.config import (
    DoubleSparsityConfig,
    parse_double_sparsity_config,
)
from sglang.srt.layers.attention.double_sparsity.selector import (
    DoubleSparsitySelector,
)
from sglang.srt.layers.attention.double_sparsity.validator import (
    validate_double_sparsity,
)

__all__ = [
    "DoubleSparsityConfig",
    "DoubleSparsitySelector",
    "parse_double_sparsity_config",
    "validate_double_sparsity",
]
