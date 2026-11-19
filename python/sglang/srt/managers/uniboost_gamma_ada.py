"""gamma-Ada: adapts the UniBoost boost rate gamma to the latency tail."""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from typing import TYPE_CHECKING, Callable, List, Optional

from sglang.srt.managers.req_types import FINISH_ABORT

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Gamma bounds (1/s), EMA weight, tail sensitivity, latency window, and the
# recorded latencies between refits.
GAMMA_ADA_MIN = 1.0
GAMMA_ADA_MAX = 200.0
GAMMA_ADA_BETA = 0.3
GAMMA_ADA_TAIL_SENSITIVITY = 4.0
GAMMA_ADA_WINDOW = 2000
GAMMA_ADA_REFIT_EVERY = 256


class GammaAdaController:
    """Every ``GAMMA_ADA_REFIT_EVERY`` recorded latencies, gamma moves by an EMA
    toward ``gamma_max / (1 + tail_sensitivity * max(t99 / t95 - 1, 0))``: a
    heavy tail pulls gamma down (more SJF), a flat tail up (more FCFS).

    With a ``step_clock`` (a step counter identical on every rank), latency counts
    steps since a sort pass first saw the request instead of wall time since
    arrival, so every rank computes the same gamma. Steps spent waiting while
    admission is closed (a full batch) before that first pass are not counted.
    """

    def __init__(
        self, gamma_init: float, *, step_clock: Optional[Callable[[], int]]
    ) -> None:
        self.gamma = min(max(gamma_init, GAMMA_ADA_MIN), GAMMA_ADA_MAX)
        self.step_clock = step_clock
        self.refit_count = 0
        self._samples = deque(maxlen=GAMMA_ADA_WINDOW)
        self._recorded_since_refit = 0

    def stamp_first_seen(self, waiting_queue: List[Req]) -> None:
        # A retracted request keeps the step it was first seen at.
        if self.step_clock is not None:
            step = self.step_clock()
            for req in waiting_queue:
                if req.uniboost_first_seen_step is None:
                    req.uniboost_first_seen_step = step

    def record_finish(self, req: Req) -> None:
        if req.finished_reason is None or isinstance(req.finished_reason, FINISH_ABORT):
            return
        if self.step_clock is not None:
            if req.uniboost_first_seen_step is None:
                return
            latency = max(self.step_clock() - req.uniboost_first_seen_step, 1)
        else:
            latency = time.perf_counter() - req.time_stats.scheduler_recv_time
            if latency <= 0.0:
                return
        self._samples.append(float(latency))
        self._recorded_since_refit += 1
        if self._recorded_since_refit >= GAMMA_ADA_REFIT_EVERY:
            self._recorded_since_refit = 0
            self._refit()

    def _refit(self) -> None:
        cuts = statistics.quantiles(self._samples, n=100, method="inclusive")
        t95, t99 = cuts[94], cuts[98]
        excess = max(t99 / t95 - 1.0, 0.0)
        target = max(
            GAMMA_ADA_MAX / (1.0 + GAMMA_ADA_TAIL_SENSITIVITY * excess),
            GAMMA_ADA_MIN,
        )
        self.gamma += GAMMA_ADA_BETA * (target - self.gamma)
        self.refit_count += 1
        logger.info(
            "UniBoost gamma-Ada refit #%d: gamma=%.4g /s (target %.4g, t99/t95=%.3f)",
            self.refit_count,
            self.gamma,
            target,
            t99 / t95,
        )
