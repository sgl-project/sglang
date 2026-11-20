"""Compiler-option compatibility shared by SAIL JIT backends."""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# HGCC selects its own PPU target and has no ptxas stage. Each option accepts
# either a separate operand or an =operand in CUDA compiler command lines.
_HGCC_DROP_WITH_ARG = frozenset(
    {"-Xptxas", "--ptxas-options", "-gencode", "--generate-code"}
)
_HGCC_DROP_PREFIX = tuple(option + "=" for option in sorted(_HGCC_DROP_WITH_ARG))
# HGCC 2.2 rejects this diagnostic switch as a driver option. Removing the
# suppression changes warnings only, unlike the math switches kept below.
_HGCC_DROP_FLAGS = frozenset({"-Wno-deprecated-declarations"})


def filter_hgcc_flags(flags: List[str]) -> List[str]:
    """Remove unsupported tuning/diagnostic options and complete operands.

    Keep all other options in order, especially --ftz/--fmad and fast-math:
    those affect numerical semantics and are supported by HGCC.
    """
    kept: List[str] = []
    dropped: List[str] = []
    skip_next = False
    for flag in flags:
        if skip_next:
            skip_next = False
            dropped.append(flag)
            continue
        if flag in _HGCC_DROP_WITH_ARG:
            skip_next = True
            dropped.append(flag)
            continue
        if flag in _HGCC_DROP_FLAGS or flag.startswith(_HGCC_DROP_PREFIX):
            dropped.append(flag)
            continue
        kept.append(flag)
    if dropped:
        logger.warning("Dropping unsupported HGCC flags: %s", dropped)
    return kept
