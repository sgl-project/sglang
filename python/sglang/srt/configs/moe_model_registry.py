"""Model eligibility, precision and prefill sizing for the DeepEP v2 adapter.

External model packages register architecture names before server-argument
validation, without adding private architecture names to SGLang core.
Only the primary architecture selects the model policy.
"""

from __future__ import annotations

from typing import Any, Callable

import msgspec


class _DeepEPv2ModelPolicy(msgspec.Struct, frozen=True):
    silu_mul_keep_fp32: bool = False
    prefill_dispatch_tokens: Callable[[Any, int], int] | None = None


_DEEPEP_V2_MODELS: dict[str, _DeepEPv2ModelPolicy] = {
    "DeepseekV3ForCausalLM": _DeepEPv2ModelPolicy(),
    "DeepseekV4ForCausalLM": _DeepEPv2ModelPolicy(),
    "Qwen3MoeForCausalLM": _DeepEPv2ModelPolicy(),
    "Glm5NextForConditionalGeneration": _DeepEPv2ModelPolicy(),
}


def register_deepep_v2_model(
    architecture: str,
    *,
    silu_mul_keep_fp32: bool = False,
    prefill_dispatch_tokens: Callable[[Any, int], int] | None = None,
) -> None:
    """Register a model architecture with validated DeepEP v2 semantics.

    ``prefill_dispatch_tokens(cfg, tokens)`` receives a read-only resolved
    configuration and the prefill-buffer ceiling. It returns the maximum
    tokens one rank dispatches, without changing scheduler/buffer settings.
    Omitting it preserves the default, unsharded ceiling.
    """

    policy = _DeepEPv2ModelPolicy(
        silu_mul_keep_fp32=silu_mul_keep_fp32,
        prefill_dispatch_tokens=prefill_dispatch_tokens,
    )
    if architecture in _DEEPEP_V2_MODELS and _DEEPEP_V2_MODELS[architecture] != policy:
        raise ValueError(f"Conflicting DeepEP v2 model policy for {architecture}")
    _DEEPEP_V2_MODELS[architecture] = policy


def _policy_of(hf_config: Any) -> _DeepEPv2ModelPolicy | None:
    architectures = getattr(hf_config, "architectures", None) or ()
    if not architectures:
        return None
    return _DEEPEP_V2_MODELS.get(architectures[0])


def model_supports_deepep_v2(hf_config: Any) -> bool:
    """Check model eligibility, independently of runtime layout and topology."""

    return _policy_of(hf_config) is not None


def model_requires_fp32_silu_mul(hf_config: Any) -> bool:
    """Whether this model opts into FP32 intermediates for DeepEP v2 MXFP8."""

    policy = _policy_of(hf_config)
    return policy is not None and policy.silu_mul_keep_fp32


def model_deepep_v2_prefill_dispatch_tokens(
    hf_config: Any, cfg: Any, default_tokens: int
) -> int:
    """Apply the primary model's dispatch sizing, or keep the buffer ceiling."""

    policy = _policy_of(hf_config)
    if policy is None or policy.prefill_dispatch_tokens is None:
        return default_tokens
    return policy.prefill_dispatch_tokens(cfg, default_tokens)
