"""MXFP4 checkpoint packing, quantization arguments, and timestep policies."""

from typing import Literal

import torch

# Policy family names are part of the msmodelslim checkpoint format.
PrecisionPolicyTarget = Literal["fa", "w4a4_linear", "w8a8_linear"]

_ALLOWED_POLICY_PRECISIONS: dict[PrecisionPolicyTarget, frozenset[str]] = {
    "fa": frozenset(("FLOAT", "MXFP4")),
    "w4a4_linear": frozenset(("W4A4", "W4A8")),
    "w8a8_linear": frozenset(("W8A8", "FLOAT")),
}


def uses_mxfp4_attention(quant_config, prefix: str) -> bool:
    description = getattr(quant_config, "quant_description", {}) or {}
    prefix = prefix.removesuffix(".impl")
    descriptor = next(
        (
            description[key]
            for key in (prefix, prefix + ".quant_type", prefix + ".fa_q.scale")
            if key in description
        ),
        None,
    )
    if descriptor is None or descriptor == "FLOAT":
        return False
    if descriptor != "MXFP4_DYNAMIC":
        raise ValueError(f"Unsupported attention descriptor for {prefix}: {descriptor}")
    return True


# Adapted from: https://gitcode.com/Ascend/MindIE-SD/blob/dev/mindiesd/quantization/layer.py
def default_mxfp4_policy(num_steps: int) -> dict[str, str]:
    """Use W4A8 for the first two and last two steps."""
    return {
        str(step): "W4A8"
        for step in range(num_steps)
        if step < 2 or step >= num_steps - 2
    }


def resolve_precision(
    description: dict,
    target: PrecisionPolicyTarget,
    default: str,
    default_policy: dict | None = None,
) -> str:
    """Resolve checkpoint step > checkpoint default > caller schedule > default."""
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context_or_none,
    )

    if target is None:
        return default

    allowed = _ALLOWED_POLICY_PRECISIONS[target]
    context = get_forward_context_or_none()
    policy = description.get("timestep_policy", {}).get(target, {})
    if not policy and not default_policy:
        return default
    if context is None:
        raise RuntimeError(f"{target} timestep policy requires a forward context")
    step = context.current_timestep
    fallback = (default_policy or {}).get(str(step), default)
    if fallback not in allowed:
        # A linear boundary schedule must not change attention precision.
        fallback = default
    precision = policy.get(str(step), policy.get("default", fallback))
    if precision not in allowed:
        raise ValueError(
            f"Unsupported {target} precision: {precision} (allowed: {sorted(allowed)})"
        )
    return precision


def mxfp4_quant_kwargs(description: dict) -> dict:
    """C7 defaults with checkpoint overrides; pair with round_mode="rint"."""
    return {
        "scale_alg": description.get("mxfp4_scale_alg", 2),
        "dst_type_max": description.get("mxfp4_dst_type_max", 7.25),
    }


def pack_fp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack numeric FP8 containers [out, in] into FP4 uint8 [out, in/2]."""
    if weight.dtype == torch.uint8:
        return weight
    if weight.dtype != torch.float8_e4m3fn or weight.shape[-1] % 2:
        raise ValueError(
            "MXFP4 weights must be packed uint8 or even-width FP8 containers"
        )
    # Convert numeric E2M1 values before the loader casts to the target dtype.
    values = weight.float()
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    codes = torch.searchsorted(magnitudes, values.abs()).clamp(max=7)
    if not torch.equal(magnitudes[codes], values.abs()):
        raise ValueError(
            "FP8 container contains values that are not representable as FP4 E2M1"
        )
    codes = (codes | (torch.signbit(values).long() << 3)).to(torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
