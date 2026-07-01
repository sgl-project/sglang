"""GC management for the scheduler process.

CPython's automatic generation-2 garbage collection is a stop-the-world pause
whose cost is O(live heap), and it triggers at arbitrary allocation sites. In
a tensor-parallel server this is a correctness-adjacent problem, not just a
latency one: a multi-second GC pause in ONE scheduler rank stalls ALL ranks,
because the paused rank stops launching kernels while the peer GPUs spin
inside a TP collective that needs its next launch. If anything in the process
slowly accumulates live objects (e.g. a leaking dependency), pause duration
grows with uptime until it exceeds the /health_generate window and the pod is
killed by its liveness probe mid-pause — which looks like a permanent silent
hang ("Server couldn't get a response from detokenizer") even though the
process would have recovered.

This was observed in production with sglang v0.5.12 on 8xB300 serving a
DeepSeek-V3-family model: a flashinfer autotuner leak (TuningConfig objects
retained by an unbounded lru_cache, see
https://github.com/flashinfer-ai/flashinfer/issues/2139) grew the scheduler
heap to ~75M tracked objects in ~6.5 hours, at which point a single gen-2
collection measured 41 s while freeing nothing. Pauses crossed the 20 s
health-check threshold after ~3.5 h and the pod was recycled several times a
day.

Strategy (all knobs env-gated):
  * Raise the gen-2 threshold so automatic full collections become rare
    (they remain as a fallback under sustained load that never goes idle).
  * On the first fully-idle tick — i.e. right after warmup — run one full
    collection and ``gc.freeze()`` the startup heap (model/module metadata,
    several million objects) into the permanent generation so subsequent
    collections never traverse it again.
  * While idle, periodically run ``gc.collect()`` so cyclic garbage is
    reclaimed at a moment when a pause cannot stall in-flight batches.

We deliberately do NOT re-freeze periodically: objects frozen while alive are
never collected if they later become cyclic garbage (e.g. evicted radix-tree
nodes, which hold parent<->child cycles), so repeated freezing would convert
ordinary churn into a permanent leak. The one-time startup freeze is safe
because the startup heap is stable for the process lifetime.
"""

from __future__ import annotations

import gc
import logging
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class SchedulerGCManager:
    def __init__(self):
        self.enabled = envs.SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT.get()
        self.idle_gc_interval_s = envs.SGLANG_SCHEDULER_IDLE_GC_INTERVAL.get()
        self._did_initial_freeze = False
        self._last_idle_gc_time = 0.0

        if not self.enabled:
            return

        gen0, gen1, gen2 = gc.get_threshold()
        new_gen2 = max(gen2, envs.SGLANG_SCHEDULER_GC_GEN2_THRESHOLD.get())
        gc.set_threshold(gen0, gen1, new_gen2)
        logger.info(
            "Scheduler GC management enabled: gen-2 threshold %d -> %d, "
            "idle GC interval %.0fs (disable with SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT=0)",
            gen2,
            new_gen2,
            self.idle_gc_interval_s,
        )

    def on_idle(self):
        """Run from the scheduler's idle housekeeping path.

        First call (right after warmup): full collect + freeze of the startup
        heap. Later calls: throttled full collect so cyclic garbage is
        reclaimed while a pause is harmless.
        """
        if not self.enabled:
            return

        now = time.perf_counter()
        if self._did_initial_freeze and (
            now < self._last_idle_gc_time + self.idle_gc_interval_s
        ):
            return

        tic = now
        collected = gc.collect()
        if not self._did_initial_freeze:
            frozen_before = gc.get_freeze_count()
            gc.freeze()
            self._did_initial_freeze = True
            logger.info(
                "Initial post-warmup GC freeze: collected %d, froze %d objects "
                "(%.3fs)",
                collected,
                gc.get_freeze_count() - frozen_before,
                time.perf_counter() - tic,
            )
        else:
            duration = time.perf_counter() - tic
            if duration > 0.1:
                logger.info(
                    "Idle GC: collected %d objects in %.3fs "
                    "(a growing duration here indicates a heap leak; "
                    "see scheduler_gc_manager.py)",
                    collected,
                    duration,
                )
        self._last_idle_gc_time = time.perf_counter()
