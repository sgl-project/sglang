# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Decode-stall diagnostics for a scheduler process, both off by default.

- ``SGLANG_LOG_GC_STALL_MS``: log every garbage-collection pass that takes at least this long.
- ``SGLANG_LOG_STALL_STACK_MS``: a daemon thread samples the main thread's frame and logs its stack once
  when the thread stays at one bytecode position for at least this long (a blocking call, a lock, or a GC
  pass that started there; the GC holds the GIL, so its sample arrives right after it ends).
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
import traceback
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class StallProbe:
    @staticmethod
    def maybe_create() -> Optional[StallProbe]:
        gc_ms = envs.SGLANG_LOG_GC_STALL_MS.get()
        stack_ms = envs.SGLANG_LOG_STALL_STACK_MS.get()
        if gc_ms <= 0 and stack_ms <= 0:
            return None
        return StallProbe(gc_threshold_s=gc_ms / 1e3, stack_threshold_s=stack_ms / 1e3)

    def __init__(self, *, gc_threshold_s: float, stack_threshold_s: float):
        self._gc_threshold_s = gc_threshold_s
        self._gc_start: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if gc_threshold_s > 0:
            gc.callbacks.append(self._on_gc)
        if stack_threshold_s > 0:
            self._thread = threading.Thread(
                target=self._sample_main_thread,
                kwargs=dict(threshold_s=stack_threshold_s),
                name="stall-probe",
                daemon=True,
            )
            self._thread.start()
        logger.info(
            "Stall probe: GC passes >= %.0f ms, main-thread stacks after %.0f ms at one position (0 = off)",
            gc_threshold_s * 1e3,
            stack_threshold_s * 1e3,
        )

    def close(self) -> None:
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_start = time.perf_counter()
            return
        if self._gc_start is None:
            return
        duration = time.perf_counter() - self._gc_start
        self._gc_start = None
        if duration >= self._gc_threshold_s:
            logger.warning(
                "GC stall: generation %s took %.1f ms (collected %s) ending at %.3f",
                info.get("generation"),
                duration * 1e3,
                info.get("collected"),
                time.time(),
            )

    def _sample_main_thread(self, *, threshold_s: float) -> None:
        main = threading.main_thread().ident
        period = threshold_s / 4  # a stall shows after 4..5 equal samples
        position, since, logged = None, 0.0, False
        while not self._stop.wait(period):
            frame = sys._current_frames().get(main)
            if frame is None:
                continue
            now = time.perf_counter()
            here = (frame.f_code, frame.f_lasti)
            if here != position:
                position, since, logged = here, now, False
            elif not logged and now - since >= threshold_s:
                logged = True
                logger.warning(
                    "Main-thread stall: at one position for %.0f ms (sampled at %.3f)\n%s",
                    (now - since) * 1e3,
                    time.time(),
                    "".join(traceback.format_stack(frame, limit=16)),
                )
