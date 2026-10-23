# SPDX-License-Identifier: Apache-2.0
"""Sends per-step request progress from each DP replica to the HTTP server."""

from __future__ import annotations

from typing import Any

import zmq
from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ProgressReporter:
    """Sends [request_id, completed, total] as JSON, dropping it if it cannot be sent now."""

    def __init__(self, context: zmq.Context, endpoint: str):
        self._socket = context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, 1000)
        self._socket.setsockopt(zmq.LINGER, 0)
        # drop instead of queueing while the HTTP server is not connected
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.connect(endpoint)
        self._warned = False

    def start(self, request_ids: list[str]) -> None:
        self.step(request_ids=request_ids, completed=0, total=0)

    def step(self, request_ids: list[str], completed: float, total: int) -> None:
        for request_id in request_ids:
            try:
                self._socket.send_json(
                    [request_id, completed, total], flags=zmq.NOBLOCK
                )
            except zmq.Again:
                pass
            except zmq.ZMQError as e:
                if not self._warned:
                    self._warned = True
                    logger.warning("Failed to send progress update: %s", e)


class ReportingTqdm(tqdm):
    """tqdm that also sends progress once per integer percent, even when disabled."""

    def __init__(
        self,
        *args: Any,
        reporter: ProgressReporter,
        request_ids: list[str],
        progress_range: tuple[float, float] = (0.0, 1.0),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._reporter = reporter
        self._request_ids = request_ids
        self._progress_range = progress_range
        self._completed = 0.0
        self._last_percent = 0
        self._iterating = False

    def update(self, n: float | None = 1) -> bool | None:
        displayed = super().update(n)
        # tqdm.__iter__ calls update() itself; __iter__ counts those steps
        if not self._iterating:
            self._advance(n or 0)
        return displayed

    def __iter__(self):
        self._iterating = True
        try:
            for obj in super().__iter__():
                yield obj
                self._advance(1)
        finally:
            self._iterating = False

    def _advance(self, n: float) -> None:
        total = self.total
        if not total or n <= 0 or self._completed >= total:
            return
        self._completed = min(self._completed + n, total)
        percent = int(self._completed * 100 // total)
        if percent > self._last_percent:
            self._last_percent = percent
            completed = self._completed
            if self._progress_range != (0.0, 1.0):
                start, end = self._progress_range
                completed = start + (end - start) * completed / total
                total = 1
            self._reporter.step(
                request_ids=self._request_ids, completed=completed, total=total
            )


_reporter: ProgressReporter | None = None


def init_progress_reporter(context: zmq.Context, endpoint: str) -> None:
    global _reporter
    try:
        _reporter = ProgressReporter(context=context, endpoint=endpoint)
    except zmq.ZMQError as e:
        logger.warning(
            "Failed to connect progress endpoint %s, progress updates disabled: %s",
            endpoint,
            e,
        )


def get_progress_reporter() -> ProgressReporter | None:
    return _reporter
