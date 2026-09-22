"""Model eligibility and MXFP8 precision policy for the DeepEP v2 adapter.

External model packages register architecture names before server-argument
validation, without adding private architecture names to SGLang core.
Only the primary architecture controls eligibility and precision.
"""

from __future__ import annotations

from typing import Any

# Membership enables DeepEP v2; the value requests FP32 SiLU intermediates.
_DEEPEP_V2_MODELS: dict[str, bool] = {
    "DeepseekV3ForCausalLM": False,
    "DeepseekV4ForCausalLM": False,
    "Qwen3MoeForCausalLM": False,
    # Qwen4-Exp (Qwen3.8-Flash-Next) hand-rolls its MoE-region comms and its
    # post-expert step is the hyper-connection combine only, so DeepEP v2's
    # folded reduction is not duplicated. Validated on block-scaled FP8, which
    # does not take the MXFP8 path -> no FP32 SiLU intermediates requested.
    "Qwen4ExpForConditionalGeneration": False,
}


def register_deepep_v2_model(
    architecture: str, *, silu_mul_keep_fp32: bool = False
) -> None:
    """Register a model architecture with validated DeepEP v2 semantics."""

    if (
        architecture in _DEEPEP_V2_MODELS
        and _DEEPEP_V2_MODELS[architecture] != silu_mul_keep_fp32
    ):
        raise ValueError(f"Conflicting DeepEP v2 precision policy for {architecture}")
    _DEEPEP_V2_MODELS[architecture] = silu_mul_keep_fp32


def model_supports_deepep_v2(hf_config: Any) -> bool:
    """Check model eligibility, independently of runtime layout and topology."""

    architectures = getattr(hf_config, "architectures", None) or ()
    architecture = architectures[0] if architectures else None
    return architecture is not None and architecture in _DEEPEP_V2_MODELS


def model_requires_fp32_silu_mul(hf_config: Any) -> bool:
    """Whether this model opts into FP32 intermediates for DeepEP v2 MXFP8."""

    architectures = getattr(hf_config, "architectures", None) or ()
    architecture = architectures[0] if architectures else None
    return architecture is not None and _DEEPEP_V2_MODELS.get(architecture, False)
