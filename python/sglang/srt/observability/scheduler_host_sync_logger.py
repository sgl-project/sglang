"""Count host-device syncs in the scheduler process by stage and call site.

With ``SGLANG_LOG_SCHEDULER_HOST_SYNCS_TARGET`` set, the scheduler turns on
``torch.cuda.set_sync_debug_mode("warn")`` at its first batch and counts every
synchronizing CUDA operation torch reports (``item()``, ``tolist()``,
``nonzero``, pageable host-to-device copies, ...) under the scheduler stage it
ran in, the forward mode of the batch being launched or processed, and its
Python call site. Every ``SGLANG_LOG_SCHEDULER_HOST_SYNCS_INTERVAL`` seconds a
``scheduler.host_syncs`` JSON record with these counts and the number of
launched batches per forward mode goes to each target, so syncs per decode step
or per extend follow by division. Explicit ``torch.cuda.synchronize()`` and
event waits are not reported by torch's sync debug mode and are not counted.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from collections import Counter
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_PROCESS_BATCH_RESULT,
    SCHEDULER_STAGE_RUN_BATCH,
    SchedulerStageMetricsRecorder,
)
from sglang.srt.utils.log_utils import create_log_targets, log_json

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

_SYNC_WARNING = "called a synchronizing CUDA operation"
_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep
_SGLANG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + os.sep
_BATCH_STAGES = (SCHEDULER_STAGE_RUN_BATCH, SCHEDULER_STAGE_PROCESS_BATCH_RESULT)
_CALL_SITE_DEPTH = 3
_MAX_CALL_SITES = 100


def _call_site(frame, depth: int = _CALL_SITE_DEPTH) -> str:
    """The innermost ``depth`` frames outside torch and warnings, innermost first."""
    parts = []
    while frame is not None and len(parts) < depth:
        filename = frame.f_code.co_filename
        if not (
            filename.startswith(_TORCH_DIR)
            or filename.endswith("warnings.py")
            or filename == __file__
        ):
            if filename.startswith(_SGLANG_DIR):
                filename = filename[len(_SGLANG_DIR) :]
            parts.append(f"{filename}:{frame.f_lineno} {frame.f_code.co_name}")
        frame = frame.f_back
    return " <- ".join(parts)


class SchedulerHostSyncLogger:
    """Process-global and never restored: the first launched batch sets torch's
    sync debug mode, adds a warnings filter and replaces ``warnings.showwarning``
    for the rest of the process."""

    def __init__(
        self,
        targets: List[str],
        dump_interval: float,
        stage_recorder: SchedulerStageMetricsRecorder,
    ):
        self.loggers = create_log_targets(targets=targets, name_prefix=__name__)
        self.dump_interval = dump_interval
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._stage_recorder = stage_recorder
        self._forward_mode = ""
        self._syncs: Counter = Counter()
        self._launched_batches: Counter = Counter()
        self._started = False
        self._show = None
        self._hook_replaced = False
        self._last_dump_time = time.monotonic()

    @staticmethod
    def maybe_create(
        enable_metrics: bool, stage_recorder: SchedulerStageMetricsRecorder
    ) -> Optional[SchedulerHostSyncLogger]:
        target = envs.SGLANG_LOG_SCHEDULER_HOST_SYNCS_TARGET.get()
        if not target:
            return None

        if not enable_metrics:
            raise ValueError(
                "SGLANG_LOG_SCHEDULER_HOST_SYNCS_TARGET is set but --enable-metrics "
                "is not active. Host-sync logging attributes syncs to scheduler "
                "stages, which require --enable-metrics."
            )

        return SchedulerHostSyncLogger(
            targets=[t.strip() for t in target.split(",") if t.strip()],
            dump_interval=envs.SGLANG_LOG_SCHEDULER_HOST_SYNCS_INTERVAL.get(),
            stage_recorder=stage_recorder,
        )

    def on_step_launch(self, batch: ScheduleBatch) -> None:
        if not self._started:
            self._start()
        self._maybe_dump()
        self._forward_mode = batch.forward_mode.name.lower()
        self._launched_batches[self._forward_mode] += 1

    def on_step_result(self, batch: ScheduleBatch) -> None:
        self._forward_mode = batch.forward_mode.name.lower()

    def _start(self) -> None:
        # "warn" rather than "error": an error could be swallowed by a broad
        # except in the scheduler, a warning always reaches this hook
        self._started = True
        warnings.filterwarnings("always", message=_SYNC_WARNING)
        show_other = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if _SYNC_WARNING in str(message):
                self._record(_call_site(sys._getframe(1)))
            else:
                show_other(message, category, filename, lineno, file, line)

        self._show = show
        warnings.showwarning = show
        torch.cuda.set_sync_debug_mode("warn")

    def _record(self, call_site: str) -> None:
        stage = self._stage_recorder.current_stage
        forward_mode = self._forward_mode if stage in _BATCH_STAGES else ""
        self._syncs[stage, forward_mode, call_site] += 1

    def _maybe_dump(self) -> None:
        now = time.monotonic()
        if now - self._last_dump_time < self.dump_interval:
            return
        if warnings.showwarning is not self._show and not self._hook_replaced:
            self._hook_replaced = True
            logger.warning(
                "warnings.showwarning was replaced after host-sync logging "
                "started; host syncs are no longer counted."
            )
        if not self._syncs and not self._launched_batches:
            self._last_dump_time = now
            return

        log_json(
            self.loggers,
            "scheduler.host_syncs",
            {
                "rank": self.rank,
                "window_s": round(now - self._last_dump_time, 3),
                "launched_batches": dict(self._launched_batches),
                "total_syncs": sum(self._syncs.values()),
                "syncs": [
                    {
                        "stage": stage,
                        "forward_mode": forward_mode,
                        "call_site": call_site,
                        "count": count,
                    }
                    for (stage, forward_mode, call_site), count in (
                        self._syncs.most_common(_MAX_CALL_SITES)
                    )
                ],
            },
        )
        self._syncs.clear()
        self._launched_batches.clear()
        self._last_dump_time = now
