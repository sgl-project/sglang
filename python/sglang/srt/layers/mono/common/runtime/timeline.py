# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/timeline.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A mono kernel's per-CTA phase stamps, saved from the server (debug only).

``ATOM_MONO_TIMELINE=<prefix>``, meant for ``--enforce-eager`` (a graph replay
runs no Python, so nothing would be saved): a kernel built with its timeline
stamps writes one [CTA][point] buffer a layer, and past the warm-up steps of each
decode token count a few steps are saved as ``<prefix>_r<rank>_s<S>_<i>.pt`` --
int64 [layer][CTA][point], ``s_memrealtime`` ticks (100 MHz), a point never
reached left 0.
"""

import collections

import torch

from sglang.srt.layers.mono.common.plan.execution import BLOCKS

WARMUP_STEPS = 20  # steps of a token count skipped before any is saved
SAVED_STEPS = 5  # steps of a token count saved


class LayerTimeline:
    def __init__(self, prefix: str, n_layers: int, points: int, rank: int, dev):
        self.prefix = prefix
        self.rank = rank
        self.stamps = torch.zeros(
            n_layers, BLOCKS, points, dtype=torch.int64, device=dev
        )
        self.steps = collections.Counter()

    def ptr(self, layer: int) -> int:
        return self.stamps[layer].data_ptr()

    def step_done(self, tokens: int) -> None:
        seen = self.steps[tokens]
        self.steps[tokens] += 1
        if WARMUP_STEPS <= seen < WARMUP_STEPS + SAVED_STEPS:
            torch.save(
                self.stamps.cpu(),
                f"{self.prefix}_r{self.rank}_s{tokens}_{seen - WARMUP_STEPS}.pt",
            )
