"""Inference options for offline ModelSlim MXFP checkpoints."""

# msmodelslim names the exported rotations after the to_q/to_k projections
# (`to_q_rot`/`to_k_rot`); earlier exports used `q_rot`/`k_rot`. The Wan weight
# mapping (configs/models/dits/wanvideo.py) accepts both spellings, so
# descriptor lookups must too -- a missed lookup leaves the rotation buffers
# unregistered and the tensors are silently skipped at load.
_ROTATION_DESCRIPTOR_NAMES = (("to_q_rot", "q_rot"), ("to_k_rot", "k_rot"))


def offline_qk_rotation_flags(description: dict, prefix: str) -> tuple[bool, bool]:
    """(q_exported, k_exported): whether ``prefix``'s Q/K rotations are FLOAT."""
    q_names, k_names = _ROTATION_DESCRIPTOR_NAMES
    return (
        any(description.get(f"{prefix}.{name}") == "FLOAT" for name in q_names),
        any(description.get(f"{prefix}.{name}") == "FLOAT" for name in k_names),
    )


def default_mxfp4_policy(num_steps: int) -> dict[str, str]:
    return {
        str(step): "W4A8"
        for step in range(num_steps)
        if step < 2 or step >= num_steps - 2
    }


def resolve_precision(
    description: dict, target: str, default: str, default_policy: dict | None = None
) -> str:
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context_or_none,
    )

    context = get_forward_context_or_none()
    policy = description.get("timestep_policy", {}).get(target, {})
    if not policy and not default_policy:
        return default
    if context is None:
        raise RuntimeError("ModelSlim timestep policy requires a forward context")
    step = context.current_timestep
    fallback = (default_policy or {}).get(str(step), default)
    precision = policy.get(str(step), policy.get("default", fallback))
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
