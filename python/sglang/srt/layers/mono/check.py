"""Check mode (``SGLANG_MONO_CHECK=1``, debugging only): the first ``SGLANG_MONO_CHECK_STEPS`` (default 4) eager
mono steps also run every routed layer's module path on copies of its inputs, and log each output's max |diff|
against it. Graph captures (and so replays) are never checked: run with --disable-cuda-graph. The module path's
cache writes land before the kernel's, which rewrites them; a step that checked is not meant to be served from.
"""

from __future__ import annotations

import logging
from typing import Mapping

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def requested() -> bool:
    return envs.SGLANG_MONO_CHECK.get()


class Checker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.limit = envs.SGLANG_MONO_CHECK_STEPS.get()
        self.steps = 0
        self._logged = False

    def begin_step(self) -> bool:
        """Whether this step's routed layers are checked (uniform over ranks: same env, same capture state)."""
        if self._logged:
            self.steps += 1
            self._logged = False
        return self.steps < self.limit and not torch.cuda.is_current_stream_capturing()

    def compare(
        self,
        layer_id: int,
        rows: int,
        pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, float]:
        """``pairs``: name -> (kernel output, module output). Synchronizes."""
        diffs = {}
        parts = []
        for name, (out, ref) in pairs.items():
            out_f, ref_f = out.float(), ref.float()
            diffs[name] = (out_f - ref_f).abs().max().item()
            parts.append(
                f"{name} max|diff| {diffs[name]:.4g} (ref max {ref_f.abs().max().item():.4g})"
            )
        logger.info(
            "[mono-check][rank %d] step %d layer %d (%d rows): %s",
            self.rank,
            self.steps,
            layer_id,
            rows,
            ", ".join(parts),
        )
        self._logged = True
        return diffs
