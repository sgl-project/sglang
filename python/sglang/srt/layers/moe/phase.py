"""Select NVFP4 activation precision for the current forward phase."""

from sglang.srt.environ import envs


def get_moe_forward_phase() -> str | None:
    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )

    if not has_forward_context():
        return None
    return get_forward_context().moe_forward_phase


def is_w4a16_enabled() -> bool:
    """Resolve the W4A16 setting for an actual forward, never for weight loading."""
    value = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.get()
    if not isinstance(value, str):
        return bool(value)
    phase = get_moe_forward_phase()
    if phase is None:
        raise RuntimeError(
            "Phase-specific SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16 requires "
            "a forward context with the prefill/decode phase."
        )
    return value == phase


def use_nvfp4_per_token_activation(quant_config, phase: str | None = None) -> bool:
    if not quant_config.use_per_token_activation:
        return False
    selected = getattr(quant_config, "per_token_activation_phase", None)
    if selected is None:
        return True
    phase = phase or get_moe_forward_phase()
    if phase is None:
        raise RuntimeError(
            "Phase-specific SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION requires "
            "a forward context with the prefill/decode phase."
        )
    return selected == phase
