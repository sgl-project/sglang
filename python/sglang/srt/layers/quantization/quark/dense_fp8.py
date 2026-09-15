"""Online w8a8 FP8 for the dense bf16 layers a quark checkpoint leaves excluded.

A quark MXFP4 checkpoint keeps a few large projections -- Qwen3.5's
``shared_expert.down_proj``, for one -- in bf16. Under ``--enable-dense-fp8`` those run
as online per-token FP8 instead, which also lets the aiter silu+mul quant kernel feed
them directly (see ``layers.quantization.fp8_silu_quant_fusion``).

Which modules qualify is the model's business, not quark's: a model registers its own
names and minimum output size through :func:`register`. :func:`linear_method_for`
returns ``None`` -- leaving the layer bf16, exactly as before -- for an unregistered
model, an ineligible layer, or the default-off flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

logger = logging.getLogger(__name__)

_POLICY_ATTR = "_dense_fp8_policy"


@dataclass
class _Policy:
    """One model's answer to "which excluded bf16 layers should go FP8?"."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    min_output_size: int = 0
    config: Optional[Fp8Config] = None


def register(
    quant_config: object,
    include: Iterable[str],
    exclude: Iterable[str] = (),
    min_output_size: int = 0,
) -> None:
    """Record a model's dense-FP8 policy on ``quant_config``, merging with any prior."""
    policy = getattr(quant_config, _POLICY_ATTR, None)
    if policy is None:
        policy = _Policy()
        setattr(quant_config, _POLICY_ATTR, policy)
    policy.include = tuple(dict.fromkeys((*policy.include, *include)))
    policy.exclude = tuple(dict.fromkeys((*policy.exclude, *exclude)))
    policy.min_output_size = max(policy.min_output_size, min_output_size)


def _flag_enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    try:
        return get_exec().kernel.enable_dense_fp8
    except ValueError:
        return False


def _eligible(policy: _Policy, prefix: str, layer: torch.nn.Module) -> bool:
    if any(name in prefix for name in policy.exclude):
        return False
    if not any(name in prefix for name in policy.include):
        return False
    n = getattr(layer, "output_size_per_partition", None) or getattr(
        layer, "output_size", None
    )
    return n is None or n >= policy.min_output_size


def linear_method_for(
    quant_config: object, prefix: str, layer: torch.nn.Module
) -> Optional[Fp8LinearMethod]:
    """Online-FP8 method for an excluded bf16 ``layer``, or None to leave it bf16."""
    policy = getattr(quant_config, _POLICY_ATTR, None)
    if policy is None or not _flag_enabled() or not _eligible(policy, prefix, layer):
        return None
    if policy.config is None:
        policy.config = Fp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )
    logger.info("[quark] routing excluded dense layer to online FP8: %s", prefix)
    return Fp8LinearMethod(policy.config)
