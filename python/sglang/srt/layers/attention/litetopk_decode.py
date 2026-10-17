"""Selector buffers that every index layer of a process shares for the opt-in
LiteTopK decode top-k (``SGLANG_OPT_LITETOPK_DECODE``)."""

from __future__ import annotations

import functools
import logging
from typing import Dict, Optional, Set

import torch

from sglang.kernels.ops.attention.litetopk_decode import (
    LiteTopKConfig,
    LiteTopKPlan,
    LiteTopKStorage,
    unsupported_reason,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.utils import print_warning_once

logger = logging.getLogger(__name__)

# DeepGEMM counts the DSA histogram of at most 128 / 32 heads tokens per request,
# so DSA target verify of more draft tokens keeps the default top-k.
DSA_MAX_NEXT_N = 4


@functools.cache
def get_litetopk_decode(config: LiteTopKConfig) -> Optional[LiteTopKDecode]:
    """The process-wide selector buffers for ``config``, or None when the flag
    is off or this GPU / DeepGEMM cannot serve it."""
    if not envs.SGLANG_OPT_LITETOPK_DECODE.get():
        return None
    reason = unsupported_reason(config)
    if reason is not None:
        logger.warning(
            "SGLANG_OPT_LITETOPK_DECODE is set, but LiteTopK %s is unsupported "
            "here (%s); the default decode top-k stays",
            config.name,
            reason,
        )
        return None
    logger.info("Decode index top-k: LiteTopK %s", config.name)
    return LiteTopKDecode(
        config=config, device=torch.device("cuda", torch.cuda.current_device())
    )


class LiteTopKDecode:
    """A plan per row count, all views of one storage (68 KiB per row) shared by
    the layers, which run in order and leave it at rest. Captured graphs keep their
    storage; a larger row count outside capture gets one of at least twice the rows
    (graph runners capture the largest batch first). Not in mem_fraction_static."""

    def __init__(self, *, config: LiteTopKConfig, device: torch.device):
        self.config = config
        self._device = device
        self._storage: Optional[LiteTopKStorage] = None
        # Streams of the eager calls on the current storage.
        self._storage_streams: Set[torch.cuda.Stream] = set()
        self._plans: Dict[int, LiteTopKPlan] = {}
        # Row counts whose plans captured graphs replay: kept with their storage.
        self._graph_rows: Set[int] = set()

    def plan(self, rows: int) -> Optional[LiteTopKPlan]:
        """The plan of ``rows`` score rows, or None (the default top-k serves the
        call) when it would need new buffers while a CUDA graph is captured."""
        capturing = torch.cuda.is_current_stream_capturing()
        plan = self._plans.get(rows)
        if plan is None:
            storage = self._storage
            if storage is None or rows > storage.max_rows:
                if capturing:
                    # Buffers allocated now would come from the graph's private pool.
                    print_warning_once(
                        f"LiteTopK decode: no buffers for {rows} rows while a CUDA "
                        "graph is captured; that graph keeps the default top-k"
                    )
                    return None
                storage = self._grow(rows)
            plan = storage.plan(rows)
            self._plans[rows] = plan
        if capturing or get_is_capture_mode():
            self._graph_rows.add(rows)
        elif plan.storage is self._storage:
            self._storage_streams.add(torch.cuda.current_stream(self._device))
        return plan

    def _grow(self, rows: int) -> LiteTopKStorage:
        # Plans of captured graphs keep the old storage; the others move to the new one.
        old = self._storage
        if old is not None and not any(
            self._plans[r].storage is old for r in self._graph_rows
        ):
            # Freed once its plans go: its eager calls may still be in flight.
            for stream in self._storage_streams:
                for buffer in (old.histogram, old.workspace):
                    buffer.record_stream(stream)
        self._plans = {r: p for r, p in self._plans.items() if r in self._graph_rows}
        max_rows = rows if old is None else max(rows, 2 * old.max_rows)
        self._storage = LiteTopKStorage(self.config, max_rows, self._device)
        self._storage_streams = set()
        logger.info(
            "LiteTopK decode: %.1f MiB of %s selector buffers for up to %d rows",
            self._storage.nbytes / (1 << 20),
            self.config.name,
            max_rows,
        )
        return self._storage
