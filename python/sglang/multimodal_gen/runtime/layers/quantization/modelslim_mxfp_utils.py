"""Inference options for offline ModelSlim MXFP checkpoints."""


def resolve_precision(description: dict, target: str, default: str) -> str:
    policy = description.get("timestep_policy", {}).get(target, {})
    if not policy:
        return default
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context,
    )

    step = get_forward_context().current_timestep
    precision = policy.get(str(step), policy.get("default", default))
    allowed = {"fa": ("FLOAT", "FP8", "MXFP4"), "w4a4_linear": ("W4A4", "W4A8")}
    if precision not in allowed[target]:
        raise ValueError(f"Unsupported ModelSlim {target} precision: {precision}")
    return precision


def mxfp4_quant_kwargs(description: dict) -> dict:
    # Match MindIE-SD f1b15b4 C7 quantization defaults.
    return {
        "scale_alg": description.get("mxfp4_scale_alg", 2),
        "dst_type_max": description.get("mxfp4_dst_type_max", 7.25),
    }
