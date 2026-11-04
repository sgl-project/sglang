"""Kimi-K3.1's DSA backends on ROCm."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_FIELDS = ("dsa_prefill_backend", "dsa_decode_backend")


def kimi_k31_dsa_backends_hip(view: Any) -> dict:
    """ROCm serves Kimi-K3.1 DSA on the Triton kernels only: they dequantize the
    reference fp8_ds_mla cache and run BF16 dots."""
    for field in _FIELDS:
        value = getattr(view, field)
        if value not in (None, "triton"):
            option = "--" + field.replace("_", "-")
            raise ValueError(f"Kimi-K3.1 supports {option} triton; got {value!r}.")
    if view.enable_hisparse:
        raise ValueError("Kimi-K3.1 does not support --enable-hisparse on ROCm.")
    logger.warning("Set DSA backends for Kimi-K3.1: prefill=triton, decode=triton.")
    return {field: "triton" for field in _FIELDS if getattr(view, field) is None}
