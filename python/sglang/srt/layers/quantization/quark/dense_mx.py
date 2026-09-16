"""Online MXFP6 for the wide dense bf16 layers an MX checkpoint leaves excluded.

A quark MXFP4 checkpoint quantizes only its routed experts, so Qwen3.5's attention
and GDN projections arrive bf16 -- 16-19% of prefill compute on MI355X. Under
``--enable-dense-mx`` those run as online MXFP6 instead.

Which modules qualify is the model's business, not quark's: a model registers its own
names, minimum output size and token threshold through :func:`register`.
:func:`linear_method_for` returns ``None`` -- leaving the layer bf16, exactly as
before -- for an unregistered model, an ineligible layer, a device without the
kernels, or the default-off flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from sglang.srt.layers.quantization.mx_dense import (
    MxDenseLinearMethod,
    mx_dense_supported,
)

logger = logging.getLogger(__name__)

_POLICY_ATTR = "_dense_mx_policy"

# Which include patterns have actually produced a converted layer. A server log
# should name them: "the flag is on" and "these projections are running MXFP6"
# are different claims, and an A/B whose arm silently converted nothing would
# otherwise read as a null result rather than a broken run.
_converted: set[str] = set()


@dataclass
class _Policy:
    """One model's answer to "which excluded bf16 layers should go MXFP6?"."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    min_output_size: int = 0
    # Forward passes narrower than this keep running bf16. MXFP6 only wins once
    # there are enough tokens to amortize quantizing the activation, and holding
    # decode at bf16 is also what stops a precision loss compounding across
    # generation steps.
    min_tokens: int = 1024


def register(
    quant_config: object,
    include: Iterable[str],
    exclude: Iterable[str] = (),
    min_output_size: int = 0,
    min_tokens: int = 1024,
) -> None:
    """Record a model's dense-MX policy on ``quant_config``, merging with any prior."""
    policy = getattr(quant_config, _POLICY_ATTR, None)
    if policy is None:
        policy = _Policy()
        setattr(quant_config, _POLICY_ATTR, policy)
    policy.include = tuple(dict.fromkeys((*policy.include, *include)))
    policy.exclude = tuple(dict.fromkeys((*policy.exclude, *exclude)))
    policy.min_output_size = max(policy.min_output_size, min_output_size)
    policy.min_tokens = max(policy.min_tokens, min_tokens)


def _flag_enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    try:
        return get_exec().kernel.enable_dense_mx
    except ValueError:
        return False


def _matched_include(policy: _Policy, prefix: str, layer: torch.nn.Module):
    """The include pattern this layer qualifies under, or None if it does not."""
    if any(name in prefix for name in policy.exclude):
        return None
    matched = next((name for name in policy.include if name in prefix), None)
    if matched is None:
        return None
    n = getattr(layer, "output_size_per_partition", None) or getattr(
        layer, "output_size", None
    )
    if n is not None and n < policy.min_output_size:
        return None
    return matched


def linear_method_for(
    quant_config: object, prefix: str, layer: torch.nn.Module
) -> Optional[MxDenseLinearMethod]:
    """Dense-MXFP6 method for an excluded bf16 ``layer``, or None to leave it bf16."""
    policy = getattr(quant_config, _POLICY_ATTR, None)
    if policy is None or not _flag_enabled() or not mx_dense_supported():
        return None
    matched = _matched_include(policy, prefix, layer)
    if matched is None:
        return None
    logger.debug("[quark] routing excluded dense layer to MXFP6: %s", prefix)
    if matched not in _converted:
        _converted.add(matched)
        logger.info(
            "Dense MXFP6 projections converted so far: %s",
            ", ".join(sorted(_converted)),
        )
    return MxDenseLinearMethod(policy.min_tokens)
