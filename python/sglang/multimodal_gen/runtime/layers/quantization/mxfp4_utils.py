"""Shared MXFP4 helpers for offline ModelSlim checkpoints on Ascend NPU.

Everything here is consumed by more than one subsystem (attention scheme,
linear schemes, the Cache-DiT precision guard, and the repack tool):

- :data:`PrecisionPolicyTarget` and the timestep precision resolver
  (:func:`resolve_precision`, :func:`default_mxfp4_policy`),
- :func:`mxfp4_quant_kwargs` (MindIE's activation-quantization defaults),
- :func:`pack_fp4_weight` used by the weight loaders and ``wan_repack.py``.

Naming note: the checkpoints are produced by the ``msmodelslim`` tool and
detected via ``--quantization modelslim``; helpers here are deliberately
named after the quantization format (MXFP4), not the tool.
"""

from typing import Literal

import torch

# A *policy family* key in the checkpoint's ``timestep_policy`` section names
# the group of layers a per-step policy governs; the entries hold precision
# *values* that family may run at:
#
#   "fa"          -> attention blocks         (FLOAT | MXFP4)
#   "w4a4_linear" -> FP4-weight linear layers (W4A4  | W4A8)
#   "w8a8_linear" -> W8A8 linear layers       (W8A8  | FLOAT)
#
# For example, ``w4a4_linear`` covers every linear whose *weights* are FP4 --
# both its W4A4 and its W4A8 steps -- while ``W4A4`` is one precision value
# that family may resolve to on a given step. The key names are part of the
# msmodelslim export contract and must not be renamed.
PrecisionPolicyTarget = Literal["fa", "w4a4_linear", "w8a8_linear"]

_ALLOWED_POLICY_PRECISIONS: dict[PrecisionPolicyTarget, frozenset[str]] = {
    "fa": frozenset(("FLOAT", "MXFP4")),
    "w4a4_linear": frozenset(("W4A4", "W4A8")),
    "w8a8_linear": frozenset(("W8A8", "FLOAT")),
}


def default_mxfp4_policy(num_steps: int) -> dict[str, str]:
    """MindIE-SD's default per-step W4A4/W4A8 schedule for single-level MXFP4.

    The first two and last two denoising steps run W4A8 (higher activation
    precision at the schedule boundaries); everything between runs W4A4. A
    40-step run uses W4A8 at zero-based steps 0, 1, 38, and 39.
    """
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
    """Resolve the effective precision for ``target`` at the current timestep.

    ``target`` is a *policy family* (see :data:`PrecisionPolicyTarget`); the
    return value is a per-step *precision* from that family's allowed set.
    Checkpoint per-step entries win over the checkpoint's family ``default``,
    which wins over ``default_policy`` (the caller's schedule, applied only to
    families the checkpoint says nothing about), which wins over ``default``.
    """
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context_or_none,
    )

    if target is None:
        # Unquantized blocks have no policy family and no per-step precision.
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
        # The caller's schedule belongs to a different policy family (e.g.
        # the linear W4A8/W4A4 boundary schedule handed to an attention
        # scheme, which only accepts FLOAT/MXFP4). Never let another
        # family's precision leak in: fall back to this family's default.
        fallback = default
    precision = policy.get(str(step), policy.get("default", fallback))
    if precision not in allowed:
        raise ValueError(
            f"Unsupported {target} precision: {precision} (allowed: {sorted(allowed)})"
        )
    return precision


def mxfp4_quant_kwargs(description: dict) -> dict:
    """Activation-quantization arguments for ``npu_dynamic_mx_quant`` calls.

    Matches MindIE-SD f1b15b4 C7 quantization defaults; checkpoints may
    override either value via ``mxfp4_scale_alg`` / ``mxfp4_dst_type_max``
    descriptors. Pair with ``round_mode="rint"`` at the call site (MindIE's
    rounding choice; the operator default differs).
    """
    return {
        "scale_alg": description.get("mxfp4_scale_alg", 2),
        "dst_type_max": description.get("mxfp4_dst_type_max", 7.25),
    }


def pack_fp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack one linear weight into two-FP4-per-byte uint8.

    Legacy msmodelslim exports store the FP4 values as a numeric
    ``float8_e4m3fn`` container ``[out, in]``; newer exports and the repack
    tool emit packed ``uint8`` ``[out, in/2]``. The loader expects packed.
    """
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
