import gc
import time

from sglang.srt.utils import freeze_gc


class IdleGCFreezer:
    """Collect, then freeze the surviving heap, once per idle period that lasts
    `idle_secs`, so later full collections skip the objects serving left alive.

    The collect keeps cyclic garbage out of the frozen set. A frozen object that
    later becomes part of a garbage cycle is not collected while it stays frozen;
    `on_cache_reset` covers the largest case, the radix tree a cache reset drops.
    """

    def __init__(self, idle_secs: float):
        self._idle_secs = idle_secs
        self._seen_forward_ct = -1
        self._idle_since = 0.0
        self._frozen = False

    def on_idle(self, forward_ct: int) -> None:
        now = time.monotonic()
        if forward_ct != self._seen_forward_ct:
            self._seen_forward_ct = forward_ct
            self._idle_since = now
            self._frozen = False
        elif not self._frozen and now - self._idle_since >= self._idle_secs:
            gc.collect()
            freeze_gc("Scheduler")
            self._frozen = True

    def on_cache_reset(self) -> None:
        """Collect the dropped tree, whose parent and child links form cycles,
        and freeze the rest again; called right after a cache reset."""
        gc.unfreeze()
        gc.collect()
        freeze_gc("Scheduler")
