"""Qwen3.5's dense-FP8 policy: which bf16 layers quark may take to online FP8, on ROCm.

Only the large excluded ``shared_expert.down_proj`` pays for itself. Routing gates,
``conv1d``, the tiny GDN ``b``/``a`` projections, embeddings and anything narrower than
:data:`_MIN_OUTPUT_SIZE` stay bf16. The names and the threshold are aiter-tuned, so they
live with the model rather than in quark.

:func:`register` is a no-op off aiter or on a non-quark checkpoint, and the policy it
hands over does nothing unless ``--enable-dense-fp8`` is set, which is not the default.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import get_bool_env_var, is_hip

_USE_AITER = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

_INCLUDE = (".shared_expert.down_proj",)
_EXCLUDE = (
    "conv1d",
    "shared_expert_gate",
    "mlp.gate",
    "in_proj_a",
    "in_proj_b",
    "in_proj_ba",
    "lm_head",
    "embed",
)
_MIN_OUTPUT_SIZE = 2048


def register(quant_config: Optional[QuantizationConfig]) -> None:
    """Hand this model's policy to quark, which decides layer by layer from there."""
    if not _USE_AITER or quant_config is None:
        return

    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

    if not isinstance(quant_config, QuarkConfig):
        return

    from sglang.srt.layers.quantization.quark import dense_fp8

    dense_fp8.register(
        quant_config,
        include=_INCLUDE,
        exclude=_EXCLUDE,
        min_output_size=_MIN_OUTPUT_SIZE,
    )
