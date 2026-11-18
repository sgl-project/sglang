import collections
import gc
import logging
import time
import types
from typing import Optional, Sequence

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_observability

logger = logging.getLogger(__name__)

__all__ = ["gc_object_counts", "configure_gc_logging", "freeze_gc", "refreeze_gc"]


def gc_object_counts():
    g0 = len(gc.get_objects(0))
    g1 = len(gc.get_objects(1))
    g2 = len(gc.get_objects(2))
    return g0, g1, g2


def configure_gc_logging(process_name: str) -> None:
    """Log every GC pass with SGLANG_LOG_GC, and warn on a pass longer than
    --gc-warning-threshold-secs."""
    log_every_gc = envs.SGLANG_LOG_GC.get()
    warning_threshold_secs = get_observability().gc_warning_threshold_secs
    if not log_every_gc and warning_threshold_secs <= 0:
        return
    start_time = 0.0

    def gc_callback(phase, info):
        nonlocal start_time
        gen = info["generation"]
        if phase == "start":
            start_time = time.perf_counter()
            if log_every_gc:
                logger.info(f"GC start: {process_name} | Generation {gen}")
            return
        duration = time.perf_counter() - start_time
        if log_every_gc:
            logger.info(
                f"GC end: {process_name} | Generation {gen} | "
                f"Duration: {duration:.4f}s | Collected: {info['collected']} | "
                f"Uncollectable: {info['uncollectable']}"
                f"{' (LONG GC)' if duration > 0.1 else ''}"
            )
        if 0 < warning_threshold_secs < duration:
            g0, g1, g2 = gc_object_counts()
            logger.warning(
                f"LONG GARBAGE COLLECTION DETECTED | {process_name} | Generation {gen} | "
                f"Duration: {duration:.4f}s | # Objects: gen0={g0}, gen1={g1}, gen2={g2} | "
                f"This may cause latency jitter. Long-lived objects keep each full pass slow; "
                f"in the scheduler, --gc-freeze-on-idle-secs freezes them."
            )

    gc.callbacks.append(gc_callback)


def freeze_gc(context: str, gc_threshold: Optional[Sequence[int]] = None):
    if gc_threshold is not None:
        gc.set_threshold(*gc_threshold)
    # Collect first, so cycles that are already garbage are freed, not frozen.
    gc.collect()
    if not logger.isEnabledFor(logging.DEBUG):
        gc.freeze()
        return
    g0_before, g1_before, g2_before = gc_object_counts()
    gc.freeze()
    g0_after, g1_after, g2_after = gc_object_counts()
    logger.debug(
        f"Freezing GC in {context} process. "
        f"gen0: {g0_before}->{g0_after}, "
        f"gen1: {g1_before}->{g1_after}, "
        f"gen2: {g2_before}->{g2_after}"
    )


def _describe(obj) -> str:
    if isinstance(obj, types.FunctionType):
        return f"function {obj.__module__}.{obj.__qualname__}"
    return type(obj).__qualname__


def refreeze_gc(context: str) -> collections.Counter:
    """Unfreeze, collect, and freeze again; return the frozen objects that are
    cyclic garbage, by type."""
    gc.collect()
    gc.unfreeze()
    saved = len(gc.garbage)
    debug = gc.get_debug()
    gc.set_debug(debug | gc.DEBUG_SAVEALL)
    try:
        gc.collect()
    finally:
        gc.set_debug(debug)
    frozen_garbage = collections.Counter(map(_describe, gc.garbage[saved:]))
    del gc.garbage[saved:]
    freeze_gc(context)
    return frozen_garbage
