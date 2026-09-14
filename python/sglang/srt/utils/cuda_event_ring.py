"""Reusable CUDA-event ring: replaces per-step ``Event()`` allocation with a
fixed set of re-recordable events."""


class ReusableEventRing:
    """Fixed ring of lazily-created, re-recordable events.

    Safety contract: a slot may be re-recorded only after every consumer of
    its previous record has waited/synchronized, so callers must size
    ``depth`` to the maximum number of records simultaneously in flight
    (plus margin).
    """

    def __init__(self, event_factory, depth: int):
        if depth < 1:
            raise ValueError(f"ReusableEventRing depth must be >= 1, got {depth}")
        self._event_factory = event_factory
        self._depth = depth
        self._ring = None
        self._slot = 0

    def next(self):
        """Return the next event in the ring (events created lazily once)."""
        if self._ring is None:
            self._ring = [self._event_factory() for _ in range(self._depth)]
        event = self._ring[self._slot]
        self._slot = (self._slot + 1) % self._depth
        return event
