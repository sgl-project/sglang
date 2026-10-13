# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/widths.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A runner's kernels, one build per step width (its rows)."""

import logging

from sglang.srt.layers.mono.common.runtime.compile import compile_kernels, scratch_bytes
from sglang.srt.layers.mono.common.runtime.consensus import tp_agree

logger = logging.getLogger("atom")


class WidthBuilds:
    """``build(rows)`` on a width's first step, its kernels (``kernels(built)``:
    ``(launcher, abi)`` pairs) compiled there without a launch: that step is a
    warmup forward before the width's graph capture, which then only records.

    Collective: every rank prepares the same width, and all agree they hold it,
    so a rank whose build failed keeps the others from launching a kernel that
    would wait on it for ever. A width refused once (a build that does not fit)
    stays refused: its steps take the original path without building again.

    A kernel with scratch refuses its width, with a warning: a persistent kernel
    needs every CTA resident at once, and one whose waves need scratch can be
    dispatched only in part (the scratch the runtime holds for the grid), its
    resident CTAs then waiting for ever on the rest."""

    def __init__(self, build, kernels, group, what: str) -> None:
        self._build, self._kernels = build, kernels
        self._group, self._what = group, what
        self._built = {}
        self._refused = set()

    def prepare(self, rows: int) -> bool:
        """Whether every rank holds the ``rows`` build."""
        if rows in self._built:
            return True
        if rows in self._refused:
            return False
        built = None
        try:
            built = self._build(rows)
            launchers = compile_kernels(self._kernels(built))
            if not self._scratch_free(launchers, rows):
                built = None
        except Exception:
            logger.exception("%s: building the %d-row kernels failed", self._what, rows)
            built = None
        if not tp_agree(built is not None, self._group):
            self._refused.add(rows)
            return False
        self._built[rows] = built
        logger.info("%s kernels built for %d rows", self._what, rows)
        return True

    def _scratch_free(self, launchers, rows: int) -> bool:
        for launcher in launchers:
            scratch = scratch_bytes(launcher)
            if scratch:
                logger.warning(
                    "%s: a %d-row kernel needs %d B of scratch a lane, which can"
                    " leave its CTAs partly resident and waiting for ever: %d-row"
                    " steps take the original path",
                    self._what,
                    rows,
                    scratch,
                    rows,
                )
                return False
        return True

    def __getitem__(self, rows: int):
        return self._built[rows]
