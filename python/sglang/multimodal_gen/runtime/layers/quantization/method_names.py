# SPDX-License-Identifier: Apache-2.0
"""Quantization method names and their deprecated aliases."""

from __future__ import annotations

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Deprecated public name -> current name.
QUANTIZATION_METHOD_ALIASES: dict[str, str] = {"kitchen_int8": "convrot_int8"}


def canonical_quantization_method(name: str) -> str:
    """Map a deprecated method name to its current name, warning once per alias."""
    canonical = QUANTIZATION_METHOD_ALIASES.get(name)
    if canonical is None:
        return name
    logger.warning_once(
        f"quantization method {name!r} is a deprecated alias of {canonical!r}"
    )
    return canonical
