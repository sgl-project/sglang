"""Shared forward temperature expression for dense TOP sampling and scoring."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

# Checks run before any sampled token leaves a forward (eager only; never during graph capture).
_PRE_EMIT_CHECKS: list[Callable[[], None]] = []


def register_pre_emit_check(check: Callable[[], None]) -> None:
    """Register a host-side check that must pass before tokens are emitted; idempotent."""
    if check not in _PRE_EMIT_CHECKS:
        _PRE_EMIT_CHECKS.append(check)


# SamplingParams.normalize serves a temperature below this as greedy decoding (top_k = 1), which TOP refuses.
_GREEDY_BELOW = 1e-6


def validate_sampling_temperature(temperature: float) -> float:
    """Validate before device execution and return the FP32 temperature value.

    The floor is checked on the FP32 value too: a request's temperature can arrive as FP32, and FP32(1e-6) is below
    1e-6, so a check on the Python float alone would admit a temperature that serving turns greedy."""
    value = float(temperature)
    if not math.isfinite(value):
        raise ValueError(f"TOP temperature must be finite, got {temperature!r}")
    try:
        fp32 = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        raise ValueError(
            f"TOP temperature must be representable as finite FP32, got {temperature!r}"
        ) from None
    if not math.isfinite(fp32):
        raise ValueError(
            f"TOP temperature must be representable as finite FP32, got {temperature!r}"
        )
    if min(value, fp32) < _GREEDY_BELOW:
        raise ValueError(
            f"TOP requires a temperature whose FP32 value is >= {_GREEDY_BELOW}, got {temperature!r} (FP32 "
            f"{fp32!r}); SGLang serves lower temperatures as greedy decoding, which TOP does not support"
        )
    return fp32


def scale_logits_for_sampling(
    logits: torch.Tensor, temperatures: torch.Tensor
) -> torch.Tensor:
    """Divide BF16 logits in place by validated FP32 per-row temperatures.

    Both engines use this operation before log-softmax. Host-side admission
    validates values; this function only checks metadata so it is graph-safe.
    """
    import torch

    if logits.dtype != torch.bfloat16 or logits.ndim != 2:
        raise ValueError("TOP temperature scaling requires 2D BF16 logits")
    if (
        temperatures.dtype != torch.float32
        or temperatures.device != logits.device
        or temperatures.shape != (logits.shape[0], 1)
    ):
        raise ValueError("TOP temperatures must be FP32 [rows, 1] on the logits device")
    if not torch.cuda.is_current_stream_capturing():
        for check in _PRE_EMIT_CHECKS:
            check()
    return logits.div_(temperatures)


def get_true_on_policy_sampling_params() -> dict:
    """Identity settings for TOP validation and internal health/warmup requests.

    Explicit neutral values also override preferred server sampling parameters;
    internal probes must not inherit user-facing transforms.
    """
    from sglang.srt.sampling.sampling_params import TOP_K_ALL

    return {
        "temperature": 1.0,
        "top_k": TOP_K_ALL,
        "top_p": 1.0,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_new_tokens": 0,
        "json_schema": None,
        "regex": None,
        "ebnf": None,
        "structural_tag": None,
        "logit_bias": None,
        "custom_params": None,
    }


def verify_true_on_policy_sampling(
    params, *, custom_logit_processor=None, return_sampling_mask=False
):
    """Allow temperature scaling without other distribution transforms.

    Call after defaults, preferred parameters and normalization are resolved,
    for every request (including each member of a batch).
    """
    validate_sampling_temperature(params.temperature)
    unsupported = [
        name
        for name, value in get_true_on_policy_sampling_params().items()
        if name != "temperature"
        and (
            getattr(params, name) != value
            if value is not None
            else bool(getattr(params, name))
        )
    ]
    if custom_logit_processor:
        unsupported.append("custom_logit_processor")
    if return_sampling_mask:
        unsupported.append("return_sampling_mask")
    if unsupported:
        raise ValueError(
            "true-on-policy supports temperature scaling with unfiltered, otherwise unmodified "
            "full-vocabulary sampling; unsupported: " + ", ".join(unsupported)
        )
